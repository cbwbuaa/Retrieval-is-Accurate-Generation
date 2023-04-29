from header import *
from dataloader import *
from models import *
from config import *
import sys
import numpy as np
import faiss
from time import time
# sys.path.append('/apdcephfs/share_916081/ponybwcao/phrase_extraction/retrieve_doc/training')
# from DPR_encoding import load_DPR, DPR_encoding_offset

def load_emb(path='embeddings.pkl'):
    with open(path, "rb") as fIn:
        stored_data = pickle.load(fIn)
    return stored_data

def save_emb(data, path='embeddings.pkl'):
    with open(path, "wb") as fOut:
        pickle.dump(data, fOut, protocol=pickle.HIGHEST_PROTOCOL)

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='wikitext103', type=str)
    parser.add_argument('--model', type=str, default='copyisallyouneed')
    parser.add_argument('--model_name', type=str, default='best_2004_400000')
    parser.add_argument('--decoding_method', type=str, default='greedy')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--mode', type=str)
    return parser.parse_args()

def encode_phrase(**args):
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'{args["root_dir"]}/ckpt/wikitext103/copyisallyouneed/{args["model_name"]}.pt')
    print(f'[!] init model over')

    local_rank = args['local_rank']
    batch_size = 512
    if local_rank == 0:
        todos = [0, 1, 2]
    elif local_rank == 1:
        todos = [3, 4, 5]
    else:
        exit('Too many gpus.')
    print(f'worker{local_rank} todo_list: {todos}')
    for idx in todos:
        phrase_result = []
        phrase_emb_result = []
        cache_phrases = []
        cache_labels = []
        input_path = f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/retrieve_doc/output/wikitext103/copyisallyouneed/ref_data/final_ref/tokenization_result_{idx}.jsonl'
        with open(input_path) as f:
            for line in tqdm(f.readlines(), desc=f'Encoding ref {idx}', disable=(local_rank != 0)):
                data = json.loads(line.strip())['results']
                phrases = [x[0] for x in data]
                label = [len(x[1]) > 0 for x in data]
                cache_phrases.append(phrases)
                cache_labels.append(label)
                if len(cache_phrases) == batch_size:
                    all_phrase, all_phrase_emb = agent.get_phrase_emb(cache_phrases, cache_labels)
                    phrase_result.extend(all_phrase)
                    phrase_emb_result.extend(all_phrase_emb)
                    cache_phrases = []
                    cache_labels = []

        if len(cache_phrases) > 0:
            all_phrase, all_phrase_emb = agent.get_phrase_emb(cache_phrases, cache_labels)
            phrase_result.extend(all_phrase)
            phrase_emb_result.extend(all_phrase_emb)
        
        phrase_emb_tensor = np.vstack(phrase_emb_result)
        
        save_emb({'phrase': phrase_result, 'embedding': phrase_emb_tensor}, f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/retrieve_doc/output/wikitext103/copyisallyouneed/phrase_index/remove_word_400000/phrase_{idx}.pkl')

def generation(**args):
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'{args["root_dir"]}/ckpt/wikitext103/copyisallyouneed/{args["model_name"]}.pt')
    print(f'[!] init model over')

    # get deliminater embeddings
    deliminater_list = [',', '.', ';', '?', '!', '\'', '`', '[', ']', '<', '>', ':', '"', '{', '}', '(', ')', '^']
    deliminater_index = [agent.model.tokenizer.encode(deliminater, add_special_tokens=False)[0] for deliminater in deliminater_list]
    deliminater_emb = agent.model.token_embeddings[deliminater_index].cpu().detach().numpy()
    
    idx = 0
    # phrase_result = load_emb(f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/retrieve_doc/output/wikitext103/copyisallyouneed/phrase_index/remove_word_400000/phrase_test.pkl')
    phrase_result = load_emb(f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/retrieve_doc/output/wikitext103/copyisallyouneed/phrase_index/remove_word_400000/phrase_{idx}.pkl')
    phrase_list, phrase_emb = phrase_result['phrase'], phrase_result['embedding']
    phrase_list += deliminater_list
    phrase_emb = np.vstack((phrase_emb, deliminater_emb))

    n, d = phrase_emb.shape
    retriever = faiss.IndexHNSWFlat(d, 32)
    # this is the default, higher is more accurate and slower to construct
    retriever.hnsw.efConstruction = 40
    retriever.hnsw.efSearch = 256
    retriever.add(phrase_emb)
    # res = faiss.StandardGpuResources()
    # retriever = faiss.index_cpu_to_gpu(res, 0, retriever)
    print(f'[!] init retriever over')

    collection = []
    with open(f'../data/wikitext103_1024/test.txt') as f:
        # collect the valid prefixes
        texts = []
        for line in tqdm(f.readlines()):
            ids = agent.model.tokenizer.encode(line, add_special_tokens=False)
            prefix, reference = ids[:32], ids[32:]
            if len(prefix) == 32:
                prefix = agent.model.tokenizer.decode(prefix)
                reference = agent.model.tokenizer.decode(reference)
                texts.append((prefix, reference))
        print(f'[!] collect {len(texts)} valid samples which have at least 32 tokens in prefix')

        for prefix, reference in tqdm(texts):
            text, candidates, time_cost = agent.retrieve_one_phrase(prefix, retriever, phrase_list, decoding_method=args["decoding_method"], top_k=0, top_p=0.95, temp=1., get_time_cost=True)
            print(prefix)
            print(text)
            print(candidates)
            exit()
            collection.append({
                'prefix': prefix, 
                'reference': reference, 
                'text': text, 
                'phrases': candidates,
                'time_cost': time_cost
            })
    return collection

    
if __name__ == "__main__":
    args = vars(parser_args())
    if args['mode'] == 'encode':
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        encode_phrase(**args)
    elif args['mode'] == 'generation':
        generation(**args)
    else:
        exit('Not implemented.')