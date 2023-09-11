from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from header import *
from torch.nn.utils.rnn import pad_sequence
import argparse
import os
import torch
import time
import pickle
import json, gc, re, random
import numpy as np
import multiprocessing as mp
sys.path.append('/apdcephfs/share_916081/ponybwcao/phrase_extraction/training')
from utils import *

def generate_mask(ids, pad_token_idx=0):
    '''generate the mask matrix of the ids, default padding token idx is 0'''
    mask = torch.ones_like(ids)
    mask[ids == pad_token_idx] = 0.
    return mask

def gpt_encoding(args):
    def inference_one_batch(batch):
        with torch.no_grad():
            input_ids = batch['input_ids'].cuda()
            mask = batch['attention_mask'].cuda()
            embeddings = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            embeddings = embeddings[:, :-1].reshape(-1, embeddings.shape[-1])
            mask = mask[:, 1:].reshape(-1).to(torch.bool)
            embeddings = embeddings[mask].cpu().numpy()
            next_toks = input_ids[:, 1:].reshape(-1)
            next_toks = next_toks[mask].cpu().tolist()
        return embeddings, next_toks

    def inference(**args):
        max_seq_len = args['max_seq_len']
        batch_size = args['batch_size']
        data = []
        embeddings, tokens = [], []
        chunk_size = 500000
        inner_idx = 0
        with open(args['data_path'].format(args['worker_idx'])) as f:
            for line in tqdm(f, disable=args['local_rank']!=0):
                document = json.loads(line)['text']
                toks = tokenizer(document, return_attention_mask=False)['input_ids']
                for st in range(0, len(toks), max_seq_len):
                    end = min(st + max_seq_len, len(toks))
                    data.append(toks[st: end])
                    if len(data) == batch_size:
                        all_gpt_ids = pad_sequence([torch.LongTensor(x) for x in data], padding_value=tokenizer.eos_token_id, batch_first=True)
                        all_gpt2_mask = generate_mask(all_gpt_ids, pad_token_idx=tokenizer.eos_token_id)
                        embed, toks = inference_one_batch({'input_ids': all_gpt_ids, 'attention_mask': all_gpt2_mask})
                        data = []
                        embeddings.append(embed)
                        tokens.extend(toks)
                        if len(tokens) >= chunk_size:
                            if args['local_rank'] == 0:
                                print(f'Dump into chunk{inner_idx}')
                            np.save(f"{args['output_dir']}/{args['model_size']}/emb_{args['worker_idx']}_{inner_idx}.npy", np.vstack(embeddings))
                            np.save(f"{args['output_dir']}/{args['model_size']}/tok_{args['worker_idx']}_{inner_idx}.npy", np.array(tokens, dtype=np.int32))
                            inner_idx += 1
                            embeddings = []
                            tokens = []
                            gc.collect()
                            torch.cuda.empty_cache()

            if len(tokens) > 0:
                if args['local_rank'] == 0:
                    print(f'Dump into chunk{inner_idx}')
                np.save(f"{args['output_dir']}/{args['model_size']}/emb_{args['worker_idx']}_{inner_idx}.npy", np.vstack(embeddings))
                np.save(f"{args['output_dir']}/{args['model_size']}/tok_{args['worker_idx']}_{inner_idx}.npy", np.array(tokens, dtype=np.int32))

    #Load GPT2
    tokenizer = GPT2Tokenizer.from_pretrained(f'/apdcephfs/share_916081/ponybwcao/PLM/GPT2-{args["model_size"]}')
    model = GPT2LMHeadModel.from_pretrained(f'/apdcephfs/share_916081/ponybwcao/PLM/GPT2-{args["model_size"]}', return_dict=True)
    model = model.cuda()
    model.eval()
    
    inference(**args)
    
def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--data_path', default='/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/knn_index/doc_split/data_chunk{}', type=str)
    parser.add_argument('--output_dir', default='/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/knn_index/embedding_tok')
    parser.add_argument('--model_size', default='small', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--max_seq_len', type=int, default=1024)
    return parser.parse_args()

def sample_tokens():
    data_dir = '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/knn_index/embedding_tok/small'
    output_dir = '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/knn_index/embedding_tok/small_sampled'
    fns = [fn for fn in os.listdir(data_dir) if fn.startswith('emb_')]
    identifiers = [re.fullmatch(f'emb_(\d+)_(\d+).npy', fn).groups() for fn in fns]
    all_emb = []
    all_tok = []
    for idx1, idx2 in tqdm(identifiers):
        emb_fn = f'{data_dir}/emb_{idx1}_{idx2}.npy'
        tok_fn = f'{data_dir}/tok_{idx1}_{idx2}.npy'
        emb = np.load(emb_fn)
        tok = np.load(tok_fn)
        random_idx = random.sample(list(range(len(emb))), int(len(emb) * 0.005))
        all_emb.append(emb[random_idx])
        all_tok.append(tok[random_idx])
    np.save(f'{output_dir}/emb.npy', np.vstack(all_emb))
    np.save(f'{output_dir}/tok.npy', np.hstack(all_tok))

def sample_tokens_mp(data_dir, output_dir, identifiers, global_idx):
    all_emb = []
    all_tok = []
    for idx1, idx2 in tqdm(identifiers, disable=global_idx!=0):
        emb_fn = f'{data_dir}/emb_{idx1}_{idx2}.npy'
        tok_fn = f'{data_dir}/tok_{idx1}_{idx2}.npy'
        emb = np.load(emb_fn)
        tok = np.load(tok_fn)
        random_idx = random.sample(list(range(len(emb))), int(len(emb) * 0.005))
        all_emb.append(emb[random_idx])
        all_tok.append(tok[random_idx])
    np.save(f'{output_dir}/emb_{global_idx}.npy', np.vstack(all_emb))
    np.save(f'{output_dir}/tok_{global_idx}.npy', np.hstack(all_tok))

def sample_tokens_runner():
    data_dir = '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/knn_index/embedding_tok/small'
    output_dir = '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/knn_index/embedding_tok/small_sampled'
    fns = [fn for fn in os.listdir(data_dir) if fn.startswith('emb_')]
    identifiers = [re.fullmatch(f'emb_(\d+)_(\d+).npy', fn).groups() for fn in fns]
    
    task_num = 10
    datasets = split_into_chunk(identifiers, task_num)
    os_context = mp.get_context('spawn')
    process_list = []
    for global_idx in range(task_num):
        p = os_context.Process(target=sample_tokens_mp, args=(data_dir, output_dir, datasets[global_idx], global_idx)) #实例化进程对象
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
    print('Begin to merge.')
    all_emb = []
    all_tok = []
    for global_idx in tqdm(range(task_num), desc='Merge'):
        emb = np.load(f'{output_dir}/emb_{global_idx}.npy')
        tok = np.load(f'{output_dir}/tok_{global_idx}.npy')
        all_emb.append(emb)
        all_tok.append(tok)
    np.save(f'{output_dir}/emb.npy', np.vstack(all_emb))
    np.save(f'{output_dir}/tok.npy', np.hstack(all_tok))
    for global_idx in tqdm(range(task_num), desc='Remove temp files'):
        os.remove(f'{output_dir}/emb_{global_idx}.npy')
        os.remove(f'{output_dir}/tok_{global_idx}.npy')
    print('Finish.')

if __name__ == '__main__':
    ## encoding
    # args = vars(parser_args())
    # torch.cuda.set_device(args['local_rank'])
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # args['worker_idx'] = args['local_rank']
    # gpt_encoding(args)

    ## sampling
    # sample_tokens()
    # sample_tokens_runner()

