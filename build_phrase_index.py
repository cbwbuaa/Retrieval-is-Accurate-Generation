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
    parser.add_argument('--model_name', type=str)
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

def build_index():
    idx = 0
    phrase_result = load_emb(f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/retrieve_doc/output/wikitext103/copyisallyouneed/phrase_index/remove_word_400000/phrase_{idx}.pkl')
    phrase_list, phrase_emb = phrase_result['phrase'], phrase_result['embedding']
    n, d = phrase_emb.shape
    index = faiss.IndexHNSWFlat(d, 32)
    # this is the default, higher is more accurate and slower to construct
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 256
    index.add(phrase_emb)
    xq = phrase_emb[:10]

    t1 = time()
    D, I = index.search(xq, 1)
    t2 = time()
    print(t2 - t1, I)

    
if __name__ == "__main__":
    args = vars(parser_args())
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args['mode'] == 'encode':
        encode_phrase(**args)
    elif args['mode'] == 'index':
        build_index()
    else:
        exit('Not implemented.')