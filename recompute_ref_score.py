import numpy as np
import json, os, collections, sys, argparse
sys.path.append('/apdcephfs/share_916081/shared_info/ponybwcao/phrase_extraction/training')
from utils import *
import torch, traceback
from tokenizer import Tokenizer
from copy import deepcopy
import nltk
from time import time
from tqdm import tqdm
from header import *
from dataloader import *
from models import *
from config import *

def parser_args():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset', default='wikitext103', type=str)
    parser.add_argument('--model', type=str, default='copyisallyouneed')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--decoding_method', type=str, default='nucleus_sampling')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--device_num', type=int)
    parser.add_argument('--device_idx', type=int)
    parser.add_argument('--nworker', type=int)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--index_type', type=str, default='pq')
    return parser.parse_args()

def find_best_ref(query_idx, agent, query_phrase_pos_list, topk_phrase_map, merged_sent, zero_phrase_embedding):
    '''
    query_idx: 要求query也来自docs, 此处query_idx即为docs中idx
    ''' 

    def load_phrase_emb(doc_idx, phrase_pos, sent_pos, embeddings, pos2tokidx_map, uniq_sent_idx_map, zero_phrase_embedding):
        uniq_sent_idx = uniq_sent_idx_map[(doc_idx, sent_pos)]
        phrase_sent_emb = embeddings[uniq_sent_idx]
        phrase_map = pos2tokidx_map[uniq_sent_idx]
        target_pos = (phrase_pos[0] - sent_pos[0], phrase_pos[1] - sent_pos[0])
        if target_pos[0] in phrase_map and target_pos[1] in phrase_map:
            real_pos = (phrase_map[target_pos[0]], phrase_map[target_pos[1]])
            phrase_emb = torch.hstack((phrase_sent_emb[real_pos[0]], phrase_sent_emb[real_pos[1]]))
        else:
            phrase_emb = zero_phrase_embedding
        return phrase_emb

    def binary_search(todo_list, target_num): # assume ascending order
        left, right = 0, len(todo_list) - 1
        mid = (left + right) // 2
        while left <= right:
            mid = (left + right) // 2
            if todo_list[mid] < target_num:
                left = mid + 1
            elif todo_list[mid] > target_num:
                right = mid - 1
            else:
                return mid
        return left

    if not topk_phrase_map:
        return [query_idx]
    query_phrases, query_phrase_pos = [], []
    for phrase, st_idx in query_phrase_pos_list:
        query_phrases.append(phrase)
        query_phrase_pos.append((st_idx, st_idx + len(phrase)))

    # cache doc, avoid dup computation
    uniq_doc_idx = set()
    uniq_prefix_end = set()
    for q_phrase, q_phrase_pos in zip(query_phrases, query_phrase_pos):
        if q_phrase not in topk_phrase_map:
            continue
        doc_refs = topk_phrase_map[q_phrase] # [scores, idxs, [pos]]
        for doc_score, doc_idx, all_pos in zip(*doc_refs):
            uniq_doc_idx.add(doc_idx)
        if q_phrase_pos != 0: # remove ref for the first phrase
            uniq_prefix_end.add(q_phrase_pos[0])
        
    if not uniq_doc_idx:
        return [query_idx]

    uniq_prefix_end = list(uniq_prefix_end) 
    prefix_map = {pos: i for i, pos in enumerate(uniq_doc_idx)}
    uniq_doc_idx = list(uniq_doc_idx)
    doc_idx_map = {doc_idx: i for i, doc_idx in enumerate(uniq_doc_idx)}
    uniq_doc = [merged_sent[x] for x in uniq_doc_idx]
    
    # encode
    q_rep, q_offsets = agent.model.encode_query_batch([merged_sent[query_idx]])
    q_rep, q_offsets = q_rep[0], q_offsets[0]
    q_st_pos, q_end_pos = [s for s, e in q_offsets], [e for s, e in q_offsets]
    all_s_rep, all_e_rep, all_offsets = agent.model.encode_doc_batch(uniq_doc, batch_size=256)
    doc_results = [(s, e, offset) for s, e, offset in zip(all_s_rep, all_e_rep, all_offsets)]

    query_embs = []
    doc_embs = []
    phrase_pair_map = {}
    for q_phrase, q_phrase_pos in zip(query_phrases, query_phrase_pos):
        if q_phrase not in topk_phrase_map:
            continue
        prefix_end_idx = binary_search(q_end_pos, q_phrase_pos[0]) - 1
        query_prefix_emb = q_rep[prefix_end_idx]
        doc_refs = topk_phrase_map[q_phrase] # [scores, idxs, [pos]]
        for doc_score, doc_idx, all_pos in zip(*doc_refs):
            s_rep, e_rep, offset = doc_results[doc_idx_map[doc_idx]]
            doc_st_pos, doc_end_pos = [s for s, e in offset[1:]], [e for s, e in offset[1:]]
            for pos in all_pos:
                st_pos, end_pos = pos, pos + len(q_phrase)
                try:
                    st_idx = doc_st_pos.index(st_pos) + 1
                    end_idx = doc_end_pos.index(end_pos) + 1
                except:
                    continue
                doc_phrase_emb = torch.hstack((s_rep[st_idx], e_rep[end_idx]))
                query_embs.append(query_prefix_emb)
                doc_embs.append(doc_phrase_emb)
                phrase_pair_map[q_phrase, q_phrase_pos, doc_idx, (st_pos, end_pos)] = len(query_embs) - 1
    query_embs = torch.vstack(query_embs)
    doc_embs = torch.vstack(doc_embs)
    phrase_scores = query_embs.unsqueeze(-2) @ doc_embs.unsqueeze(-1)
    phrase_scores = phrase_scores.view(-1).cpu().tolist()

    # update vocab
    phrase_score_map = collections.defaultdict(list)
    for key, index in phrase_pair_map.items():
        q_phrase, q_phrase_pos, doc_idx, doc_phrase_pos = key
        score = phrase_scores[index]
        phrase_score_map[q_phrase, q_phrase_pos].append((doc_idx, doc_phrase_pos, score))
    phrase_scores = [query_idx]
    for key, value in phrase_score_map.items():
        q_phrase, q_phrase_pos = key
        all_doc_idxs = [x[0] for x in value]
        all_doc_phrase_pos = [x[1] for x in value]
        all_phrase_score = [x[2] for x in value]
        phrase_scores.append((q_phrase, q_phrase_pos, all_doc_idxs, all_doc_phrase_pos, all_phrase_score))
    return phrase_scores

def worker(**args):
    worker_id = args['local_rank']
    nworker = args['nworker']

    fp16 = False
    if fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # corpus
    corpus_path = '/apdcephfs/share_916081/ponybwcao/Copyisallyouneed/data/wikitext103_1024/base_data_128.txt'
    merged_sent, doc_labels = load_base_data(corpus_path, quiet=(worker_id != 0))
    datasets = split_into_chunk(list(range(len(merged_sent))), nworker)
    
    # topk refs
    topk = 5
    doc_ref_f = open(f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/retrieve_doc/output/wikitext103/copyisallyouneed/docs/doc_ref_8_split/refs/top{topk}_doc_refs_{worker_id}.jsonl', 'r')
    candidate_f = open(f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/retrieve_doc/output/wikitext103/copyisallyouneed/docs/doc_ref_8_split/candidates/candidates_{worker_id}.jsonl', 'r')

    # model
    if worker_id == 0:
        print(f"Loading encoder...")
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/bak/wikitext103/copyisallyouneed/best_new_inbatch_prebatch256_indoc2_100000.pt')
    if worker_id == 0:
        print(f'[!] init model over')

    output_path = f'tmp.jsonl'
    chunk_size = 10 # 10000
    all_result = []
    topk_refs = []
    zero_phrase_embedding = torch.zeros(768).cuda()
    for doc_idx in tqdm(datasets[worker_id], desc=f'Worker{worker_id} process', disable=(worker_id != 0)):
        query = merged_sent[doc_idx]
        if len(topk_refs) == 0:
            query_candidates = load_lines_chunk(candidate_f, chunk_size)
            query_candidates = [json.loads(x) for x in query_candidates]
            topk_refs = load_lines_chunk(doc_ref_f, chunk_size)
            topk_refs = [json.loads(x) for x in topk_refs]
        # try:
        query_phrase_pos_list = query_candidates.pop(0)
        topk_phrase_map = topk_refs.pop(0)
        phrase_score_map = find_best_ref(doc_idx, agent, query_phrase_pos_list, topk_phrase_map, merged_sent, zero_phrase_embedding)
        # except Exception as e:
        #     phrase_score_map = []
        #     with open(f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/retrieve_doc/output/wikitext103/{mode}/log/worker{worker_id}.log', 'a') as f_error:
        #         f_error.write(f'error at worker{worker_id}, doc {doc_idx}\n')
        #         f_error.write(f'--{query}--\n')
        #         f_error.write(f'--{query_phrase_pos_list}--\n')
        #         f_error.write('str(Exception):\t' + str(Exception) + '\n')
        #         f_error.write('repr(e):\t' + repr(e) + '\n')
        #         # f_error.write('traceback.print_exc():\t' + traceback.print_exc() + '\n')
        #         # f_error.write('traceback.format_exc():\n' + traceback.format_exc() + '\n')
        #         f_error.write('-' * 20 + '\n')
        all_result.append(phrase_score_map)
        if len(all_result) == chunk_size:
            with open(output_path, 'a') as f:
                for result in all_result:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            all_result = []

    doc_ref_f.close()
    candidate_f.close()
    if len(all_result) > 0:
        with open(output_path, 'a') as f:
            for result in all_result:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    args = vars(parser_args())
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    
    worker(**args)
