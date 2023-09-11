from header import *
from dataloader import *
from models import *
from config import *
import sys, collections, random, math
import numpy as np
import faiss
from time import time
# sys.path.append('/apdcephfs/share_916081/shared_info/ponybwcao/phrase_extraction/training')
# from tokenizer import Tokenizer
# from utils import *
import faiss.contrib.torch_utils
from copy import deepcopy
from QA_test import load_QA_dataset
# import nltk
# from nltk import word_tokenize
# nltk.download('punkt')

def load_emb(path='embeddings.pkl'):
    with open(path, "rb") as fIn:
        stored_data = pickle.load(fIn)
    return stored_data

def save_emb(data, path='embeddings.pkl'):
    with open(path, "wb") as fOut:
        pickle.dump(data, fOut, protocol=pickle.HIGHEST_PROTOCOL)

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='wikipedia', type=str)
    parser.add_argument('--testset', type=str)
    parser.add_argument('--model', type=str, default='copyisallyouneed')
    parser.add_argument('--training_data_dir', type=str, default='/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_all_phrase_ref_check_valid')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--model_size', type=str, default='small')
    parser.add_argument('--decoding_method', type=str, default='nucleus_sampling')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--device_num', type=int)
    parser.add_argument('--device_idx', type=int)
    parser.add_argument('--chunk_idx', type=int)
    parser.add_argument('--phrase_dim', type=int, default=128)
    parser.add_argument('--nworker', type=int)
    parser.add_argument('--cluster_idx', type=int)
    parser.add_argument('--topk', type=int, default=512)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--index_type', type=str, default='pq')
    parser.add_argument('--random_initialize', type=bool, default=False)
    return parser.parse_args()

def encode_phrase(**args):
    args['mode'] = 'test'
    local_rank = args['local_rank']

    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'../ckpt/wikitext103/copyisallyouneed/train_pipeline/{args["model_name"]}.pt')
    if local_rank == 0:
        print(f'[!] init model over')

    base_data = {}
    with open(f'../data/wikitext103_1024/base_data_128.txt') as f:
        for line in tqdm(f.readlines(), disable=(local_rank != 0)):
            line = line.strip().split('\t')
            chunk = ' '.join(line[:-1])
            id_label = line[-1].strip()
            if id_label:
                base_data[id_label] = chunk
    if local_rank == 0:
        print(f'[!] load base data over')

    phrase_result = []
    phrase_emb_result = []
    phrase_pos = []
    cache_phrases = []
    cache_doc_idxs = []

    worker_idx = args['worker_idx']
    print('Begin worker', worker_idx)
    input_f = open(f'/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_candidates/candidates_{worker_idx}.jsonl')

    output_dir = f'../phrase_index/{args["model_name"]}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunk_size = 10000
    batch_size = 512
    input_lines = []
    cnt = 0
    while True:
        if len(input_lines) == 0:
            input_lines = load_lines_chunk(input_f, chunk_size)
            if len(input_lines) == 0:
                print(worker_idx, 'done.')
                break
            else:
                if len(phrase_emb_result) > 0:
                    phrase_emb_npy = np.vstack(phrase_emb_result)
                    np.save(f'{output_dir}/phrase_{worker_idx}_{cnt // chunk_size - 1}.npy', phrase_emb_npy)
                    save_emb({'phrase': phrase_result, 'idx': phrase_pos}, f'{output_dir}/text_idx_list_{worker_idx}_{cnt // chunk_size - 1}.pkl')
                    save_emb({phrase_pos[i]: i for i in range(len(phrase_pos))}, f'{output_dir}/idx2emb_map_{worker_idx}_{cnt // chunk_size - 1}.pkl')
                    phrase_emb_result = []
                    phrase_result = []
                    phrase_pos = []
                cnt += chunk_size
                if local_rank == 0:
                    print(f'\tProcessing {cnt - chunk_size} - {cnt}')
                if os.path.exists(f'{output_dir}/phrase_{worker_idx}_{cnt // chunk_size - 1}.npy'):
                    input_lines = []
                    continue
        line = input_lines.pop(0)
        line = json.loads(line.strip())
        data, doc_idx = line['candidates'], line['index']
        cache_phrases.append(data)
        cache_doc_idxs.append(doc_idx)
        if len(cache_phrases) == batch_size:
            all_phrase, all_phrase_emb, all_phrase_pos = agent.get_phrase_emb(cache_phrases, cache_doc_idxs, base_data)
            phrase_result.extend(all_phrase)
            phrase_emb_result.extend(all_phrase_emb)
            phrase_pos.extend(all_phrase_pos)
            cache_phrases = []
            cache_labels = []
            cache_doc_idxs = []
            
    input_f.close()
    if len(cache_phrases) > 0:
        all_phrase, all_phrase_emb, all_phrase_pos = agent.get_phrase_emb(cache_phrases, cache_doc_idxs, base_data)
        phrase_result.extend(all_phrase)
        phrase_emb_result.extend(all_phrase_emb)
        phrase_pos.extend(all_phrase_pos)
        cache_phrases = []
        cache_labels = []
        cache_doc_idxs = []

    phrase_emb_npy = np.vstack(phrase_emb_result)
    np.save(f'{output_dir}/phrase_{worker_idx}_{cnt // chunk_size - 1}.npy', phrase_emb_npy)
    save_emb({'phrase': phrase_result, 'idx': phrase_pos}, f'{output_dir}/text_idx_list_{worker_idx}_{cnt // chunk_size - 1}.pkl')
    save_emb({phrase_pos[i]: i for i in range(len(phrase_pos))}, f'{output_dir}/idx2emb_map_{worker_idx}_{cnt // chunk_size - 1}.pkl')

def merge_result():
    input_dir = f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/phrase_index/{args["model_name"]}'
    all_fns = os.listdir(input_dir)
    phrase_num = []
    # emb
    all_phrase_emb = []
    for device_id in tqdm(range(8), desc=f'Merging phrase emb'):
        fns = [fn for fn in all_fns if fn.startswith(f'phrase_{device_id}_')]
        fns.sort()
        fns = [os.path.join(input_dir, fn) for fn in fns]
        phrase_emb = []
        phrase_num.append([])
        for fn in fns:
            phrase_emb_ = np.load(fn)
            phrase_emb.append(phrase_emb_)
            phrase_num[-1].append(phrase_emb_.shape[0])
        phrase_emb = np.vstack(phrase_emb)
        all_phrase_emb.append(phrase_emb)
    all_phrase_emb = np.vstack(all_phrase_emb)
    np.save(f'{input_dir}/phrase_emb.npy', all_phrase_emb)
    
    # idx2emb map
    cur_st_idx = 0
    global_map = {}
    for device_id in tqdm(range(8), desc=f'Merging idx2emb map'):
        fns = [fn for fn in all_fns if fn.startswith(f'idx2emb_map_{device_id}_')]
        fns.sort()
        fns = [os.path.join(input_dir, fn) for fn in fns]
        assert len(fns) == len(phrase_num[device_id])
        for fn, num in zip(fns, phrase_num[device_id]):
            idx2emb_map = load_emb(fn)
            for k, v in idx2emb_map.items():
                global_map[k] = cur_st_idx + v
            cur_st_idx += num
    save_emb(global_map, f'{input_dir}/idx2emb_map.pkl')

    # text idx
    global_text_list = []
    global_idx_list = []
    for device_id in tqdm(range(8), desc=f'Merging text_pos map'):
        fns = [fn for fn in all_fns if fn.startswith(f'text_idx_list_{device_id}_')]
        fns.sort()
        fns = [os.path.join(input_dir, fn) for fn in fns]
        for fn in fns:
            text_idx = load_emb(fn)
            global_text_list.extend(text_idx['phrase'])
            global_idx_list.extend(text_idx['idx'])
    save_emb({'phrase': global_text_list, 'idx': global_idx_list}, f'{input_dir}/text_idx_list.pkl')

def build_index(gpu_list=None, dim=128):
    ngpu = faiss.get_num_gpus()
    if not gpu_list:
        gpu_list = list(range(ngpu))
    gpu_resources = []
    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        # res.setTempMemory(tempmem)
        gpu_resources.append(res)
    vres = faiss.GpuResourcesVector()
    vdev = faiss.Int32Vector()
    for i, gpu_idx in enumerate(gpu_list):
        vdev.push_back(gpu_idx)
        vres.push_back(gpu_resources[gpu_idx])
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    index = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, faiss.IndexFlatIP(dim), co)
    return index
    
def generation(**args):
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'/apdcephfs/share_916081/ponybwcao/tmp/copyisallyouneed_v2/ckpt/wikipedia/copyisallyouneed/train/{args["model_name"]}.pt')
    # agent.load_model(f'{args["root_dir"]}/ckpt/wikitext103/copyisallyouneed/train/{args["model_name"]}.pt')
    agent.model.eval()
    print(f'[!] init model over')

    phrase_list = load_emb('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/phrase_embedding_index/PCA_phrase_merged.pkl')

    token_embs = agent.model.dim_proj(agent.model.token_embeddings).cpu().detach()
    phrase_embedding = np.load('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/phrase_embedding_index/PCA_emb_merged.npy')
    phrase_embedding = torch.from_numpy(phrase_embedding)
    retriever = build_index()
    # retriever.add(token_embs)
    # retriever.add(phrase_embedding)
    retriever.add(torch.vstack([token_embs, phrase_embedding]))
    print(f'[!] init retriever over')

    tok_norm = torch.norm(token_embs, dim=-1)
    phrase_norm = torch.norm(phrase_embedding, dim=-1)
    print('token norm:', torch.mean(tok_norm), torch.max(tok_norm))
    print('phrase norm:', torch.mean(phrase_norm), torch.max(phrase_norm))

    prefix_length = 128
    test_num = 5
    collection = []
    with open(f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/minipile/raw/test.jsonl') as f:
        # collect the valid prefixes
        texts = []
        for line in tqdm(f.readlines()):
            line = json.loads(line)['text']
            ids = agent.model.tokenizer.encode(line, add_special_tokens=False)
            prefix, reference = ids[:prefix_length], ids[prefix_length:]
            if len(prefix) == prefix_length:
                prefix = agent.model.tokenizer.decode(prefix)
                reference = agent.model.tokenizer.decode(reference[:args['max_gen_len']])
                texts.append((prefix, reference))
                if len(texts) == test_num:
                    break
        print(f'[!] collect {len(texts)} valid samples which have at least {prefix_length} tokens in prefix')
        t1 = time()
        for prefix, reference in tqdm(texts):
            text, candidates, time_cost, phrase_cnt, tok_cnt = agent.retrieve_one_phrase(prefix, retriever, phrase_list, decoding_method=args["decoding_method"], top_k=0, top_p=0.95, temp=1., get_time_cost=True)
            print(prefix)
            # print('-'*30)
            # print(reference)
            print('-'*30)
            print(text)
            print('-'*30)
            print(candidates)
            print(phrase_cnt, tok_cnt)
            print(time_cost)
            collection.append({
                'prefix': prefix, 
                'reference': reference, 
                'text': text, 
                'phrases': candidates,
                'time_cost': time_cost
            })
        exit()
    return collection

def clustering_phrase_multiproc(**args):    
    data_dir = '../phrase_index/best_prebatch_neg0_pretrain40w_1000000'
    phrase_emb = None
    for i in tqdm(range(8), desc='Loading phrase embbeding', disable=args['local_rank'] != 0):
        path = f'{data_dir}/phrase_{i}.npy'
        data = np.load(path)
        if phrase_emb is None:
            phrase_emb = data
        else:
            phrase_emb = np.vstack((phrase_emb, data))
    
    text_idx = load_emb(f'{data_dir}/global_text_idx_list.pkl')
    phrase_text = text_idx['phrase']
    phrase_idx = text_idx['idx']
    
    # phrase_emb = phrase_emb[:10000]
    # phrase_text = phrase_text[:10000]
    # phrase_idx = phrase_idx[:10000]
    assert phrase_emb.shape[0] == len(phrase_text) == len(phrase_idx)

    phrase_map = collections.defaultdict(list)
    for i, phrase in enumerate(phrase_text):
        phrase_map[phrase].append(i)

    all_phrases = list(phrase_map.keys())
    worker_datasets = split_into_chunk(all_phrases, args['device_num'])
    phrases = split_into_chunk(worker_datasets[args['device_idx']], args['nworker'])[args['local_rank']]

    d = phrase_emb.shape[1]
    new_phrase_emb = WithMutex(None)
    new_phrase_text = WithMutex([]) # no need to save idx again
    new_idx2emb_map = WithMutex({}) # for training

    thread_num = 4
    chunk_size = 10000
    chunk_idx = WithMutex(0)
    datasets = partition(phrases, thread_num)

    def func(ctx: dict):
        task = ctx.get('task')
        bar: WithMutex = ctx.get('bar')
        worker_id = ctx.get('worker_id')
        keys = ctx.get('datasets')[worker_id]

        for k in keys:
            v = phrase_map[k]
            emb = phrase_emb[v]
            text = [phrase_text[x] for x in v]
            doc_idx = [phrase_idx[x] for x in v]
            if len(v) == 1:
                add_phrase_emb = emb
                add_phrase_text = [k]
                idx_map_inner_pos = [0]
            else:
                ncentroids = int(np.log10(len(v)) + 1) # floor
                kmeans = faiss.Kmeans(d, ncentroids, niter=10, verbose=False, gpu=False, min_points_per_centroid=1)
                kmeans.train(emb)
                # D, I = kmeans.index.search(emb, 1)
                # add_phrase_emb = kmeans.centroids
                # add_phrase_text = [k for _ in range(ncentroids)]
                # idx_map_inner_pos = [i[0] for i in I]
                index = faiss.IndexFlatL2(d)
                index.add(emb)
                D, I = index.search(kmeans.centroids, 1)
                real_centroids_idx = [i[0] for i in I]
                real_centroids = emb[real_centroids_idx]

                centroids_index = faiss.IndexFlatL2(d)
                centroids_index.add(real_centroids)
                D, I = centroids_index.search(emb, 1)
                idx_map_inner_pos = [i[0] for i in I]
                add_phrase_emb = real_centroids
                add_phrase_text = [k for _ in range(ncentroids)]
            
            with bar.mutex:
                bar.obj()

            with new_phrase_emb.mutex:
                cur_pos = len(new_phrase_text.obj)
                if new_phrase_emb.obj is None:
                    new_phrase_emb.obj = add_phrase_emb
                else:
                    new_phrase_emb.obj = np.vstack((new_phrase_emb.obj, add_phrase_emb))
                new_phrase_text.obj.extend(add_phrase_text)
                for idx, pos in zip(doc_idx, idx_map_inner_pos):
                    new_idx2emb_map.obj[idx] = cur_pos + pos
                if new_phrase_emb.obj.shape[0] >= chunk_size:
                    np.save(f'{data_dir}/cluster_phrase_{args["device_idx"]}_{args["local_rank"]}_{chunk_idx.obj}.npy', new_phrase_emb.obj)
                    save_emb(new_idx2emb_map.obj, f'{data_dir}/cluster_idx2emb_map_{args["device_idx"]}_{args["local_rank"]}_{chunk_idx.obj}.pkl')
                    save_emb(new_phrase_text.obj, f'{data_dir}/cluster_text_list_{args["device_idx"]}_{args["local_rank"]}_{chunk_idx.obj}.pkl')
                    chunk_idx.obj += 1
                    new_phrase_emb.obj = None

    ParalledTask.create(f"Clustering phrase embeddings")\
                .set_nworker(thread_num)\
                .set_worker_args({'datasets': datasets})\
                .set_worker_func(func)\
                .set_progress_goal(len(phrases))\
                .execute()

    if new_phrase_emb.obj.shape[0] > 0:
        np.save(f'{data_dir}/cluster_phrase_{args["device_idx"]}_{args["local_rank"]}_{chunk_idx.obj}.npy', new_phrase_emb.obj)
        save_emb(new_idx2emb_map.obj, f'{data_dir}/cluster_idx2emb_map_{args["device_idx"]}_{args["local_rank"]}_{chunk_idx.obj}.pkl')
        save_emb(new_phrase_text.obj, f'{data_dir}/cluster_text_list_{args["device_idx"]}_{args["local_rank"]}_{chunk_idx.obj}.pkl')

def merge_cluster_result():
    input_dir = f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/phrase_index/best_prebatch_neg0_pretrain40w_1000000'
    fns = os.listdir(input_dir)
    phrase_num = []
    # emb
    all_phrase_emb = []
    for device_id in range(4):
        phrase_num.append([])
        for worker_id in range(1):
            phrase_emb = []
            emb_path_prefix = f'cluster_phrase_{device_id}_{worker_id}_'
            emb_fns = [fn for fn in fns if fn.startswith(emb_path_prefix) and fn.endswith('.npy')]
            emb_fns.sort()
            emb_fns = [os.path.join(input_dir, fn) for fn in emb_fns]
            for emb_fn in tqdm(emb_fns, desc=f'Merging phrase emb{device_id}-{worker_id}'):
                phrase_emb_ = np.load(emb_fn)
                phrase_emb.append(phrase_emb_)
            phrase_emb = np.vstack(phrase_emb) # device - worker
            phrase_num[-1].append(phrase_emb.shape[0])
            all_phrase_emb.append(phrase_emb)
    all_phrase_emb = np.vstack(all_phrase_emb)
    np.save(f'{input_dir}/cluster_phrase.npy', all_phrase_emb)
    
    # idx2emb map
    cur_st_idx = 0
    global_map = {}
    for device_id in range(4):
        for worker_id in range(1):
            map_path = f'{input_dir}/cluster_idx2emb_map_{device_id}_{worker_id}.pkl'
            idx2emb_map = load_emb(map_path)
            for k, v in tqdm(idx2emb_map.items(), desc=f'processing map{device_id}-{worker_id}'):
                global_map[k] = cur_st_idx + v
            cur_st_idx += phrase_num[device_id][worker_id]
    save_emb(global_map, f'{input_dir}/cluster_idx2emb_map.pkl')

    global_text_list = []
    for device_id in range(4):
        for worker_id in range(1):
            text = load_emb(f'{input_dir}/cluster_text_list_{device_id}_{worker_id}.pkl')
            global_text_list.extend(text)
    save_emb(global_text_list, f'{input_dir}/cluster_text_list.pkl')

def train_index_opq(num_clusters=256, code_size=96, hnsw=False, cuda=True):
    data_dir = '/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/phrase_index/best_prebatch_neg0_pretrain40w_1000000'
    trained_index_path = f'{data_dir}/IP_opq_index_{num_clusters}_{code_size}.faiss'
    phrase_emb = np.load(f'{data_dir}/cluster_phrase.npy')

    ds = phrase_emb.shape[1]
    quantizer = faiss.IndexFlatIP(ds)

    opq_matrix = faiss.OPQMatrix(ds, code_size)
    opq_matrix.niter = 10
    sub_index = faiss.IndexIVFPQ(quantizer, ds, num_clusters, code_size, 8, faiss.METRIC_INNER_PRODUCT)
    start_index = faiss.IndexPreTransform(opq_matrix, sub_index)

    start_index.verbose = False
    if cuda:
        # Convert to GPU index
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        # start_index = faiss.index_cpu_to_gpu(res, 0, start_index, co)
        start_index.verbose = False
        
        index_ivf = faiss.extract_index_ivf(start_index)
        index_ivf.nprobe = 256
        quantizer = index_ivf.quantizer
        quantizer_gpu = faiss.index_cpu_to_all_gpus(quantizer)
        index_ivf.quantizer = quantizer_gpu

        # Train on GPU and back to CPU
        phrase_emb = torch.tensor(phrase_emb).cuda()
        start_index.train(phrase_emb)
        start_index.add(phrase_emb)
        start_index = faiss.index_gpu_to_cpu(start_index)
        D, I = start_index.search(phrase_emb[:5], 5)
        print(D)
    else:
        start_index.train(phrase_emb)

    index_ivf = faiss.extract_index_ivf(start_index)
    index_ivf.make_direct_map()
    index_ivf.set_direct_map_type(faiss.DirectMap.Hashtable)

    faiss.write_index(start_index, trained_index_path)

def train_quantizer():
    data_dir = '/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/phrase_index/best_prebatch_neg0_pretrain40w_1000000'
    phrase_emb = np.load(f'{data_dir}/cluster_phrase.npy')
    d = phrase_emb.shape[-1]  # data dimension
    cs = 96  # code size (bytes)

    pq = faiss.ProductQuantizer(d, cs, 8)
    pq.train(phrase_emb)

    # encode 
    codes = pq.compute_codes(phrase_emb)
    print(codes.shape)
    np.save(f'{data_dir}/cluster_phrase_pq_encoded.npy', codes)
    # decode
    x2 = pq.decode(codes)

    # compute reconstruction error
    avg_relative_error = ((phrase_emb - x2)**2).sum() / (phrase_emb ** 2).sum()
    print(avg_relative_error)

def train_index(method, cuda=True):
    data_dir = f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/phrase_index/{args["model_name"]}'
    phrase_emb = np.load(f'{data_dir}/phrase_emb.npy')

    # phrase_emb = None
    # for i in tqdm(range(8), desc='Loading phrase embbeding'):
    #     path = f'{data_dir}/phrase_{i}.npy'
    #     data = np.load(path)
    #     if phrase_emb is None:
    #         phrase_emb = data
    #     else:
    #         phrase_emb = np.vstack((phrase_emb, data))
    n = phrase_emb.shape[0]
    print('n:', n)
    print("Building", method)
    d = phrase_emb.shape[1]
    if method == 'pq':
        method = 'pq96s'
        # m = 16                                   # number of subquantizers
        # n_bits = 8                               # bits allocated per subquantizer
        # index = faiss.IndexPQ(d, m, n_bits)        # Create the index
        index = faiss.index_factory(d, 'PQ96', faiss.METRIC_INNER_PRODUCT)
        index.train(phrase_emb)                       # Training
        index.add(phrase_emb)                          # Populate the index
        # t1 = time()
        # D, I = index.search(phrase_emb[:20], 3) 
        # print(D, I)
        # print(time() - t1)

    elif method == 'flat':
        index = faiss.IndexFlatIP(d)
        # index = faiss.index_factory(d, 'Flat', faiss.METRIC_INNER_PRODUCT)
        # phrase_emb = torch.nn.functional.normalize(phrase_emb, p=2, dim=-1)
        # faiss.normalize_L2(phrase_emb)
        index.add(phrase_emb)

        # t1 = time()
        # D, I = index.search(phrase_emb[:20], 3) 
        # print(D, I)
        # print(time() - t1)

    elif method == 'hnsw_flat':
        method = 'hnsw32_flat'
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        # 40 is the default, higher is more accurate and slower to
        # construct
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 256
        # faiss.normalize_L2(phrase_emb)
        index.add(phrase_emb)

        # index.hnsw.search_bounded_queue = True
        # t1 = time()
        # D, I = index.search(phrase_emb[:20], 3) 
        # print(D, I)
        # print(time() - t1)

    elif method == 'hnsw_sq':
        # also set M so that the vectors and links both use 128 bytes per
        # entry (total 256 bytes)
        index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, 16, faiss.METRIC_INNER_PRODUCT)

        print("training")
        # training for the scalar quantizer
        index.train(phrase_emb)

        # this is the default, higher is more accurate and slower to
        # construct
        index.hnsw.efConstruction = 40
        index.add(phrase_emb)

        index.hnsw.efSearch = 256
        t1 = time()
        D, I = index.search(phrase_emb[:20], 3) 
        print(D, I)
        print(time() - t1)

    elif method == 'hnsw_pq':
        method = 'hnsw32pq96'
        # hnsw pq does not support IP!!!
        index = faiss.index_factory(d, 'HNSW32,PQ96', faiss.METRIC_INNER_PRODUCT)
        
        # index = faiss.IndexHNSWPQ(d, 32, 16, faiss.METRIC_INNER_PRODUCT)
        # faiss.normalize_L2(phrase_emb)
        print("training")
        # training for the scalar quantizer
        index.train(phrase_emb)
        # this is the default, higher is more accurate and slower to
        # construct
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 256
        index.add(phrase_emb)

        # t1 = time()
        # D, I = index.search(phrase_emb[:20], 3) 
        # print(D, I)
        # print(time() - t1)

    elif method == 'ivf_pq':
        nlist = int(math.sqrt(n))
        m = 16
        bits_per_index = 8
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits_per_index)
        index.train(phrase_emb)
        index.add(phrase_emb)
    
    elif method == 'opq32_Rflat':
        index = faiss.index_factory(d, "OPQ32,PQ32x4fs,RFlat", faiss.METRIC_INNER_PRODUCT)
        faiss.omp_set_num_threads(32)
        index.train(phrase_emb)
        index.add(phrase_emb)
    
    elif method == 'pca':
        mat = faiss.PCAMatrix(768, 80)
        mat.train(phrase_emb)
        assert mat.is_trained
        tr = mat.apply(mt)
        # print this to show that the magnitude of tr's columns is decreasing
        print (tr ** 2).sum(0)
    else:
        index = faiss.index_factory(768, "OPQ64_256,IVF65536_HNSW32,PQ64x4fsr", faiss.METRIC_INNER_PRODUCT)
        # index = faiss.index_factory(768, "OPQ32_256,IVF65536_HNSW32,PQ32x4fsr")
        index.train(phrase_emb)
        index.add(phrase_emb)

        t1 = time()
        D, I = index.search(phrase_emb[:20], 3)
        print(D, I)
        print(time() - t1)
    # index = faiss.index_gpu_to_cpu(index)
    trained_index_path = f'{data_dir}/IP_{method}_index.faiss'
    faiss.write_index(index, trained_index_path)

def test():
    data_dir = '/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/phrase_index/best_prebatch_neg0_pretrain40w_1000000'

    name = 'ivf_pq'
    trained_index_path = f'{data_dir}/IP_{name}_index.faiss'
    start_index = faiss.read_index(trained_index_path)
    if 'hnsw' in name:
        start_index.hnsw.efSearch = 256
    phrase_emb = np.load(f'{data_dir}/cluster_phrase.npy')
    phrase_emb = torch.tensor(phrase_emb).cuda()
    
    index_ivf = faiss.extract_index_ivf(start_index)
    index_ivf.make_direct_map()
    index_ivf.set_direct_map_type(faiss.DirectMap.Hashtable)

    res = faiss.StandardGpuResources()
    if 'ivf' in name:
        start_index.nprobe = 128
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        gpu_index = faiss.index_cpu_to_gpu(res, 0, start_index, co)
    else:
        start_index = faiss.index_cpu_to_gpu(res, 0, start_index)
    

    # n = 10
    # k = 5
    # d = 768

    # D, I, R = start_index.search_and_reconstruct(phrase_emb[:n], k)
    # D, I, R = start_index.search(phrase_emb[:n], k)
    # # print(I)
    # # print(R)
    # print(D.shape, I.shape, R.shape)

    # faiss.normalize_L2(phrase_emb)
    topk = 128
    t1 = time()
    # xq = torch.tensor(phrase_emb[:10])
    for _ in range(10):
        D, I = gpu_index.search(phrase_emb[_*600:_*600 + 600], 100)
        emb = []
        for i in I:
            for x in i:
                emb.append(start_index.reconstruct(x.item()))
        emb = torch.vstack(emb).cuda()
        # D, I = start_index.search(phrase_emb[:600], 100)
    print((time() - t1) / 10)
    t1 = time()
    # xq = torch.tensor(phrase_emb[:10])
    for _ in range(10):
        D, I, R = start_index.search_and_reconstruct(phrase_emb[_*600:_*600 + 600].cpu(), 100)
        I = I.cuda()
        R = R.cuda()
        # D, I = start_index.search(phrase_emb[:600], 100)
    print((time() - t1) / 10)
    # print(sum([i in I[i] for i in range(phrase_emb.shape[0])]) / phrase_emb.shape[0])

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def viterbi_forward(one_choice_result):
        tok_type, tok_result, phrase_result = one_choice_result['type'], one_choice_result['tok_result'], one_choice_result['phrase_result']
        phrase2tok_idx_map, tok2phrase_idx_map = one_choice_result['index_map']
        tok_num = len(tok_type)
        # init
        dp = [0 for _ in range(tok_num)]
        dp[0] = tok_result[0][1] # token prob
        if tok_type[0] == 'S':
            for x in phrase_result[0]:
                if len(x[0].split()) == 1:
                    dp[0] += x[1] # phrase prob
        dp[0] = np.log(dp[0])
        for i in range(1, tok_num): # cur token
            dp[i] = np.exp(dp[i - 1] + np.log(tok_result[i][1]))
            if tok_type[i] in ['S', 'E']: 
                cur_phrase_idx = tok2phrase_idx_map[i]
                for prev_phrase_idx in range(cur_phrase_idx + 1): # generate phrase
                    prev_tok_idx = phrase2tok_idx_map[prev_phrase_idx]
                    st, _ = prev_tok_idx
                    for x in phrase_result[st]:
                        if len(x[0].split()) == cur_phrase_idx - prev_phrase_idx + 1:
                            if st == 0:
                                dp[i] += x[1]
                            else:
                                dp[i] += np.exp(dp[st - 1] + np.log(x[1]))
            dp[i] = np.log(dp[i])
        score = dp[-1] / tok_num
        return score

def general_viterbi_forward(one_choice_result):
    tok_result, phrase_result = one_choice_result['token_result'], one_choice_result['phrase_result']
    tok_num = len(tok_result)
    # init
    dp = [0 for _ in range(tok_num)]
    # dp[0] = tok_result[0][1] # token prob
    # if tok_type[0] == 'S':
    #     for x in phrase_result[0]:
    #         if len(x[0].split()) == 1:
    #             dp[0] += x[1] # phrase prob
    # dp[0] = np.log(dp[0])
    for tok_idx in range(tok_num): # cur token
        if tok_idx == 0:
            dp[tok_idx] = tok_result[tok_idx][1]
        else:
            dp[tok_idx] = np.exp(dp[tok_idx - 1] + np.log(tok_result[tok_idx][1]))
        for prev_tok_idx in range(tok_idx + 1): # generate phrase
            for phrase, prob, tok_num in phrase_result[prev_tok_idx]:
                if tok_num == tok_idx - prev_tok_idx + 1:
                    if prev_tok_idx == 0:
                        dp[tok_idx] += prob
                    else:
                        dp[tok_idx] += np.exp(dp[prev_tok_idx - 1] + np.log(prob))
        dp[tok_idx] = np.log(dp[tok_idx] + 1e-10)
    score = dp[-1] / tok_num
    return score

@torch.no_grad()
def _OBQA_test(**args):
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'/apdcephfs/share_916081/ponybwcao/tmp/copyisallyouneed_v2/ckpt/wikipedia/copyisallyouneed/train/best_0906_EM1_small_bs2_temp2.0_ce_loss_lr1e-4_50000.pt')
    agent.model.eval()
    print(f'[!] init model over')

    phrase_list = load_emb('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/phrase_embedding_index/PCA_phrase_merged.pkl')

    token_embs = agent.model.dim_proj(agent.model.token_embeddings).cpu().detach()
    phrase_embedding = np.load('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/phrase_embedding_index/PCA_emb_merged.npy')
    phrase_embedding = torch.from_numpy(phrase_embedding)
    retriever = build_index()
    # retriever.add(token_embs)
    retriever.add(phrase_embedding)
    print(f'[!] init retriever over')

    # processing
    batch_size = 64
    topk = args['topk']
    stride = 256
    start_time = time()

    dataset_name = args['testset']
    dataset = load_QA_dataset(dataset_name)
    print(f'load {len(dataset)} examples from {dataset_name}')

    results = []
    for item in tqdm(dataset, desc=dataset_name):
        question = item['question']
        choices = item['choices']
        gt = item['label']

        gpt_ids = []
        all_phrase_pred_idx = []
        all_tok_type = []
        all_ids = []
        all_query = []
        all_suffix_pos = []
        pred_results = [collections.defaultdict(list) for _ in range(len(choices))]

        question_toks = agent.model.tokenizer(question, add_special_tokens=False)['input_ids']
        for choice_idx, choice in enumerate(choices):
            phrases = word_tokenize(choice)
            query = question
            phrase_st_pos = []
            for phrase in phrases:
                phrase_st_pos.append(len(query) + 1)
                query += ' ' + phrase
            all_query.append(query)

            phrases_ = [' ' + x for x in phrases]
            phrase_ids = agent.model.tokenizer(phrases_, add_special_tokens=False)['input_ids']
            tokens = []
            tokens.extend(question_toks)
            phrase_pred_idx = []
            phrase2tok_idx_map = {}
            tok2phrase_idx_map = {}
            tok_num_cnt = 0
            for i, ids in enumerate(phrase_ids):
                phrase2tok_idx_map[i] = (tok_num_cnt, tok_num_cnt + len(ids))
                for tok_idx in range(tok_num_cnt, tok_num_cnt + len(ids)):
                    tok2phrase_idx_map[tok_idx] = i
                tok_num_cnt += len(ids)
                all_ids.extend(ids)
                if len(ids) == 1:
                    all_tok_type.append('S') # single
                else:
                    all_tok_type.extend(['B'] + ['I' for _ in range(1, len(ids) - 1)] + ['E'])
                all_suffix_pos.extend([(len(all_query) - 1, phrase_st_pos[i])] + [(len(all_query) - 1, None) for _ in range(1, len(ids))])
                if len(tokens) + len(ids) > 1024:
                    gpt_ids.append(tokens)
                    all_phrase_pred_idx.append(phrase_pred_idx)
                    tokens = tokens[stride:]
                    phrase_pred_idx = []
                phrase_pred_idx.extend(range(len(tokens), len(tokens) + len(ids)))
                tokens.extend(ids)
            gpt_ids.append(tokens)
            all_phrase_pred_idx.append(phrase_pred_idx)
            pred_results[choice_idx]['index_map'].extend([phrase2tok_idx_map, tok2phrase_idx_map])

        gpt2_ids = pad_sequence([torch.LongTensor(i) for i in gpt_ids], padding_value=agent.model.tokenizer.eos_token_id, batch_first=True).cuda()
        max_length = gpt2_ids.shape[1]
        phrase_pred_idx = []
        for i, pred_idx in enumerate(all_phrase_pred_idx):
            for idx_ in pred_idx:
                phrase_pred_idx.append(idx_ + max_length * i)
        phrase_pred_idx = torch.LongTensor(phrase_pred_idx) - 1 # predicted by prev token
        gpt2_mask = generate_mask(gpt2_ids, pad_token_idx=agent.model.tokenizer.eos_token_id).cuda()
        with torch.no_grad():
            query_reps = agent.model.model(input_ids=gpt2_ids, attention_mask=gpt2_mask, output_hidden_states=True).hidden_states[-1]
            if agent.model.dim_proj is not None:
                query_reps = agent.model.dim_proj(query_reps)
            query_reps = query_reps.view(-1, query_reps.shape[-1])[phrase_pred_idx]
        D, I = index.search(query_reps.cpu(), topk)
        D = D.cuda()
        all_token_ip = torch.matmul(query_reps, token_embeddings.t())

        for topk_index, topk_IP, token_ip, token_id, token_type, suffix_pos_, q_rep in zip(I, D, all_token_ip, all_ids, all_tok_type, all_suffix_pos, query_reps):
            logits = torch.cat([topk_IP, token_ip], dim=0)
            probs = torch.softmax(logits, dim=-1)
            pred_results[suffix_pos_[0]]['type'].append(token_type)
            tok = agent.model.tokenizer.convert_ids_to_tokens(token_id)
            pred_results[suffix_pos_[0]]['tok_result'].append((tok, probs[topk + token_id].item()))
            if token_type in ['S', 'B']: # start token
                suffix_ = all_query[suffix_pos_[0]][suffix_pos_[1]:] + ' ' # ' ' for last word
            else: # inner token
                pred_results[suffix_pos_[0]]['phrase_result'].append([])
                continue
            phrase_result = []
            for i, (idx_, prob_) in enumerate(zip(topk_index, probs[: topk])):
                pred_phrase = index_phrase[idx_]
                pred_doc_idx = index_pos[idx_][0].split(',')[0]
                # print(pred_phrase)
                if suffix_.startswith(pred_phrase + ' '):
                    phrase_result.append((pred_phrase, prob_.item()))
            pred_results[suffix_pos_[0]]['phrase_result'].append(phrase_result)
        
        all_score = []
        for x in pred_results:
            score = viterbi_forward(x)
            all_score.append(score)
       
        pred = int(np.argmax(all_score))
        # output_f.write(json.dumps({'pred': pred, 'score': all_score, **item}, ensure_ascii=False) + '\n')
        results.append(gt == pred)
    print(sum(results) / len(results))

@torch.no_grad()
def OBQA_test(**args):
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'/apdcephfs/share_916081/ponybwcao/tmp/copyisallyouneed_v2/ckpt/wikipedia/copyisallyouneed/train/{args["model_name"]}.pt')
    agent.model.eval()
    print(f'[!] init model over')

    phrase_list = load_emb('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/phrase_embedding_index/PCA_phrase_merged.pkl')

    token_embs = agent.model.dim_proj(agent.model.token_embeddings).detach()
    phrase_embedding = np.load('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/phrase_embedding_index/PCA_emb_merged.npy')
    phrase_embedding = torch.from_numpy(phrase_embedding)
    retriever = build_index()
    # retriever.add(token_embs)
    retriever.add(phrase_embedding)
    # retriever.add(torch.vstack([token_embs, phrase_embedding]))
    print(f'[!] init retriever over')

    # processing
    batch_size = 64
    topk = args['topk']
    stride = 256
    start_time = time()

    dataset_name = args['testset']
    dataset = load_QA_dataset(dataset_name)
    print(f'load {len(dataset)} examples from {dataset_name}')

    results = []
    phrase_cnt = 0
    for item in tqdm(dataset, desc=dataset_name):
        question = item['question']
        choices = item['choices']
        gt = item['label']
        text_list = [question + ' ' + x for x in choices]

        encoded_dict = agent.model.tokenizer(text_list, return_attention_mask=False, return_offsets_mapping=True)
        ids = encoded_dict['input_ids']
        all_st_char = pad_sequence([torch.LongTensor([y[0] for y in x]) for x in encoded_dict['offset_mapping']], padding_value=-1, batch_first=True)
        all_length = [len(x) for x in ids]
        all_gpt_ids = pad_sequence([torch.LongTensor(x) for x in ids], padding_value=agent.model.tokenizer.eos_token_id, batch_first=True)
        all_gpt2_mask = pad_sequence([torch.ones(len(x)).to(torch.long) for x in ids], padding_value=0, batch_first=True)
        output = agent.model.model(input_ids=all_gpt_ids.cuda(), attention_mask=all_gpt2_mask.cuda(), output_hidden_states=True)
        hidden = output['hidden_states'][-1] # (B, S, H)
        hidden = agent.model.dim_proj(hidden)

        query = hidden[:, :-1].reshape(-1, hidden.shape[-1])
        query_mask = all_gpt2_mask[:, 1:].reshape(-1).to(torch.bool)
        query = query[query_mask]
        label = all_gpt_ids[:, 1:].reshape(-1)[query_mask]
        st_char_map = all_st_char[:, 1:].reshape(-1)[query_mask]
        
        D, I = retriever.search(query.cpu(), topk)
        token_ip = torch.matmul(query, token_embs.t())
        logits = torch.hstack([token_ip, D.cuda()])
        probs = torch.softmax(logits, dim=-1).cpu()

        pred_results = []
        cur_idx = 0
        for instance_idx, length in enumerate(all_length):
            st_idx = cur_idx
            end_idx = st_idx + length - 1
            input_text = text_list[instance_idx]
            sub_probs = probs[st_idx: end_idx]
            sub_label = label[st_idx: end_idx]
            sub_st_char_map = st_char_map[st_idx: end_idx].tolist()
            sub_I = I[st_idx: end_idx].tolist()
            one_result = {'token_result': [], 'phrase_result': []}
            for tok_idx in range(len(sub_probs)):
                prob_ = sub_probs[tok_idx]
                label_ = sub_label[tok_idx]
                st_char_ = sub_st_char_map[tok_idx]
                suffix_ = input_text[st_char_:]
                has_white_space = suffix_.startswith(' ')
                suffix_ = suffix_.lstrip() + ' '
                retrieved_phrases = [phrase_list[x] for x in sub_I[tok_idx]]

                tok_prob = prob_[label_]
                phrase_result = []
                for phrase_idx, phrase in enumerate(retrieved_phrases):
                    if suffix_.startswith(phrase + ' '):
                        phrase_cnt += 1
                        phrase_prob_ = prob_[len(token_embs) + phrase_idx].item()
                        if has_white_space:
                            phrase_tokens = agent.model.tokenizer(' ' + phrase, return_attention_mask=False)['input_ids']
                        else:
                            phrase_tokens = agent.model.tokenizer(phrase, return_attention_mask=False)['input_ids']
                        tok_num = len(phrase_tokens)
                        phrase_result.append([phrase, phrase_prob_, tok_num])
                one_result['token_result'].append([label_, tok_prob.item()])
                one_result['phrase_result'].append(phrase_result)
            pred_results.append(one_result)
            cur_idx = end_idx
        all_score = []
        for x in pred_results:
            score = general_viterbi_forward(x)
            all_score.append(score)
        # print(all_score)
        pred = int(np.argmax(all_score))
        # output_f.write(json.dumps({'pred': pred, 'score': all_score, **item}, ensure_ascii=False) + '\n')
        results.append(gt == pred)
    print(f'{phrase_cnt} phrase retrieved.')
    print(sum(results) / len(results))

@torch.no_grad()
def OBQA_test_gpt2(**args):
    # model
    args['mode'] = 'test'
    args['model'] = 'gpt2'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/ckpt/wikitext103/gpt2/train/{args["model_name"]}.pt')
    agent.model.model.eval()
    print(f'[!] init model over')
    # processing
    stride = 256
    start_time = time()

    # data_path = '/apdcephfs/share_916081/shared_info/ponybwcao/phrase_extraction/data/QA/truthful_qa/mc_data.jsonl'
    data_path = '/apdcephfs/share_916081/shared_info/ponybwcao/phrase_extraction/data/QA/hellaswag/data_validation.jsonl'
    dataset = []
    with open(data_path) as f:
        for line in tqdm(f):
            dataset.append(json.loads(line))
    print(f'[!] collect {len(dataset)} samples.')
    # output_f = open('/apdcephfs/share_916081/shared_info/ponybwcao/phrase_extraction/data/QA/truthful_qa/mc_data_pred_gpt2.jsonl', 'w')
    output_f = open('/apdcephfs/share_916081/shared_info/ponybwcao/phrase_extraction/data/QA/hellaswag/validation_pred.jsonl', 'w')
    results = []
    for item in tqdm(dataset):
        # question = item['question']
        # choices = item['mc1_targets']['choices']
        # gt = int(np.argmax(item['mc1_targets']["labels"]))
        question = item['ctx']
        choices = item['endings']
        gt = int(item['label'])
        gpt_ids = []
        all_phrase_pred_idx = []
        question_toks = agent.model.tokenizer(question, add_special_tokens=False)['input_ids']
        for choice_idx, choice in enumerate(choices):
            phrases = word_tokenize(choice)
            phrases_ = [' ' + x for x in phrases]
            phrase_ids = agent.model.tokenizer(phrases_, add_special_tokens=False)['input_ids']
            tokens = []
            tokens.extend(question_toks)
            phrase_pred_idx = []
            for i, ids in enumerate(phrase_ids):
                if len(tokens) + len(ids) > 1024:
                    gpt_ids.append(tokens)
                    all_phrase_pred_idx.append(phrase_pred_idx)
                    tokens = tokens[stride:]
                    phrase_pred_idx = []
                phrase_pred_idx.extend(range(len(tokens), len(tokens) + len(ids)))
                tokens.extend(ids)
            gpt_ids.append(tokens)
            all_phrase_pred_idx.append(phrase_pred_idx)

        gpt2_ids = pad_sequence([torch.LongTensor(i) for i in gpt_ids], padding_value=agent.model.tokenizer.eos_token_id, batch_first=True).cuda()
        gpt2_mask = generate_mask(gpt2_ids, pad_token_idx=agent.model.tokenizer.eos_token_id).cuda()

        gen_logits = agent.model.model(input_ids=gpt2_ids, attention_mask=gpt2_mask).logits
        all_score = []
        for i, pred_idx in enumerate(all_phrase_pred_idx):
            pred_idx = torch.LongTensor(pred_idx)
            shift_logits = gen_logits[i][pred_idx - 1]
            shift_labels = gpt2_ids[i][pred_idx]
            loss = agent.model.gen_loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            ppl = math.exp(loss.item())
            all_score.append(ppl)
        pred = int(np.argmin(all_score))
        output_f.write(json.dumps({'pred': pred, 'score': all_score, **item}, ensure_ascii=False) + '\n')
        results.append(gt == pred)
    print(sum(results) / len(results))

@torch.no_grad()
def regenerate_ref_retrieve(**args):
    def process_batch(all_gpt_ids, all_document, all_global_suffix_st_char, all_length):
        total_tok_cnt_, match_phrase_cnt_ = 0, 0
        gpt_ids = pad_sequence([torch.LongTensor(x) for x in all_gpt_ids], padding_value=tokenizer.eos_token_id, batch_first=True)
        # gpt2_mask = generate_mask(gpt_ids, pad_token_idx=tokenizer.eos_token_id)
        gpt2_mask = pad_sequence([torch.ones(len(x)).to(torch.long) for x in all_gpt_ids], padding_value=0, batch_first=True)
        gpt_reps = model(input_ids=gpt_ids.cuda(), attention_mask=gpt2_mask.cuda(), output_hidden_states=True).hidden_states[-1]
        
        gpt2_mask = gpt2_mask[:, 1:].reshape(-1).to(torch.bool)
        gpt_reps = gpt_reps[:, :-1, :].reshape(-1, gpt_reps.shape[-1])
        if dim_proj is not None:
            gpt_reps = dim_proj(gpt_reps).cpu()
        gpt_reps = gpt_reps[gpt2_mask]
        assert sum(all_length) - len(all_length) == gpt_reps.shape[0], print(sum(all_length) - len(all_length), gpt_reps.shape[0])
        D, I = retriever.search(gpt_reps, topk)
        cur_idx = 0
        all_hard_negs = []
        all_valid_phrases = []
        for instance_idx, length in enumerate(all_length):
            st_idx = cur_idx
            end_idx = st_idx + length - 1
            document = all_document[instance_idx]
            global_suffix_st_char = all_global_suffix_st_char[instance_idx][1:]
            assert len(global_suffix_st_char) == length - 1
            valid_phrases = []
            hard_negs = [[]]
            for tok_idx in range(st_idx, end_idx):
                topk_pred = I[tok_idx].tolist()
                suffix_st_char = global_suffix_st_char[tok_idx - st_idx]
                if suffix_st_char != -1:
                    total_tok_cnt_ += 1
                    suffix_ = document[suffix_st_char: suffix_st_char + max_suffix_length].lstrip() + ' '
                    for candidate_idx in topk_pred:
                        cand_phrase_ = phrase_list[candidate_idx]
                        if suffix_.startswith(cand_phrase_ + ' '):
                            valid_phrases.append([cand_phrase_, tok_idx - st_idx + 1, -1, candidate_idx])
                            match_phrase_cnt_ += 1
                            break
                hard_negs.append([candidate_idx for candidate_idx in topk_pred])
                # hard_negs.append([(candidate_idx, phrase_list[candidate_idx]) for candidate_idx in topk_pred])
            all_valid_phrases.append(valid_phrases)
            all_hard_negs.append(hard_negs)
            cur_idx += length - 1
        return all_valid_phrases, all_hard_negs, total_tok_cnt_, match_phrase_cnt_

    def merge_valid_phrases(ori_valid_phrases, new_valid_phrases): # new phrase has higher priority
        assert len(ori_valid_phrases) == len(new_valid_phrases)
        merged_valid_phrases = []
        for ori_list, new_list in zip(ori_valid_phrases, new_valid_phrases):
            cache_st_idx = {x[1]: 1 for x in new_list}
            merged_list = new_list
            for x in ori_list:
                phrase, st, end, ref = x
                if st not in cache_st_idx:
                    merged_list.append(x)
                    cache_st_idx[st] = 1
            merged_valid_phrases.append(merged_list)
        return merged_valid_phrases

    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    if args['model_size'] == 'small':
        agent.load_model(f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/ckpt/wikipedia/copyisallyouneed/train/best_0901_small_bs4_temp2.0_focal_lr1e-4_100000.pt')
    elif args['model_size'] == 'medium':
        agent.load_model(f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/ckpt/wikipedia/copyisallyouneed/train/best_0901_medium_bs2_temp2.0_focal_lr1e-4_150000.pt')
    agent.model.eval()
    model = agent.model.model
    tokenizer = agent.model.tokenizer
    dim_proj = agent.model.dim_proj
    print(f'[!] init model over')

    phrase_list = load_emb('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/phrase_embedding_index/PCA_phrase_merged.pkl')

    phrase_embedding = torch.from_numpy(np.load('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/phrase_embedding_index/PCA_emb_merged.npy'))
    retriever = build_index([1, 2, 3])
    retriever.add(phrase_embedding)
    print(f'[!] init retriever over')

    # processing
    batch_size = 52
    topk = 128
    max_suffix_length = 72
    total_tok_cnt = 0
    match_phrase_cnt = 0
    merged_phrase_cnt = 0

    chunk_idx = args['chunk_idx']
    data_path = f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/minipile/match_result_tok/train_chunk{chunk_idx}_tok_{args["model_size"]}'
    output_path = f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/minipile/match_result_tok/train_chunk{chunk_idx}_tok_{args["model_size"]}_EM'
    done_num = sum([1 for _ in open(output_path)])
    worker_cnt = 0
    with open(data_path) as data_f, open(output_path, 'a') as output_f:
        results = collections.defaultdict(int)
        all_gpt_ids, ori_valid_phrases, all_document, all_global_suffix_st_char, all_length = [], [], [], [], []
        for line in tqdm(data_f):
            worker_cnt += 1
            if worker_cnt <= done_num:
                continue
            gpt_ids, valid_phrases, document, global_suffix_st_char = json.loads(line)
            all_gpt_ids.append(gpt_ids)
            ori_valid_phrases.append(valid_phrases)
            all_length.append(len(gpt_ids))
            all_document.append(document)
            all_global_suffix_st_char.append(global_suffix_st_char)
            if len(all_gpt_ids) == batch_size:
                all_valid_phrases, all_hard_negs, total_tok_cnt_, match_phrase_cnt_ = process_batch(all_gpt_ids, all_document, all_global_suffix_st_char, all_length)
                merged_valid_phrases = merge_valid_phrases(ori_valid_phrases, all_valid_phrases)
                for instance_idx in range(len(all_gpt_ids)):
                    output_f.write(json.dumps([all_gpt_ids[instance_idx], merged_valid_phrases[instance_idx], all_document[instance_idx], all_global_suffix_st_char[instance_idx], all_hard_negs[instance_idx]], ensure_ascii=False) + '\n')
                match_phrase_cnt += match_phrase_cnt_
                merged_phrase_cnt += sum([len(x) for x in merged_valid_phrases])
                total_tok_cnt += total_tok_cnt_
                all_gpt_ids, ori_valid_phrases, all_document, all_global_suffix_st_char, all_length = [], [], [], [], []
                # print(match_phrase_cnt, total_tok_cnt, merged_phrase_cnt)

        if len(all_gpt_ids) > 0:
            all_valid_phrases, all_hard_negs, total_tok_cnt_, match_phrase_cnt_ = process_batch(all_gpt_ids, all_document, all_global_suffix_st_char, all_length)
            merged_valid_phrases = merge_valid_phrases(ori_valid_phrases, all_valid_phrases)
            for instance_idx in range(len(all_gpt_ids)):
                output_f.write(json.dumps([all_gpt_ids[instance_idx], merged_valid_phrases[instance_idx], all_document[instance_idx], all_global_suffix_st_char[instance_idx], all_hard_negs[instance_idx]], ensure_ascii=False) + '\n')
            match_phrase_cnt += match_phrase_cnt_
            merged_phrase_cnt += len(merged_valid_phrases)
            total_tok_cnt += total_tok_cnt_
        print(match_phrase_cnt, match_phrase_cnt, total_tok_cnt, match_phrase_cnt / total_tok_cnt, match_phrase_cnt / total_tok_cnt)

def _regenerate_ref_rank(**args):
    def choose_reference(query, score_vocab, merged_score_vocab, ref_vocab, doc_labels, merged_sent, length_penalty=1.0):
        word_st_pos = []
        cur_idx = 0
        query_words = query.split()
        for tok in query_words:
            cur_idx = query.find(tok, cur_idx)
            word_st_pos.append(cur_idx)
            cur_idx += len(tok)
        references = []
        for word, st_pos in zip(query_words, word_st_pos):
            related_phrase = [(v / len(k[0].split()) ** length_penalty, *k) for k, v in score_vocab.items() if k[1] == st_pos]
            if not related_phrase:
                references.append([word, []])
            else:
                related_phrase.sort()
                best_phrase = related_phrase[-1][1:]
                all_refs = [(scores, doc_refs) for scores, doc_refs in zip(merged_score_vocab[best_phrase], ref_vocab[best_phrase])]
                all_refs.sort()
                # print(word, related_phrase)
                # pos
                best_doc_idx, best_doc_pos = all_refs[-1][-1]
                assert best_doc_pos[1] - best_doc_pos[0] == len(best_phrase[0])
                # ref_doc = merged_sent[best_doc_idx]
                # ref_doc = ref_doc[:best_doc_pos[0]] + '**' + best_phrase[0] + '**' + ref_doc[best_doc_pos[1]:]
                # print(ref_doc)

                best_ref = [doc_labels[best_doc_idx], best_doc_pos[0], best_doc_pos[1]]
                references.append([word, best_ref])
        return references

    nworker = args['nworker']
    worker_id = args['local_rank']
    corpus_path = '/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/data/wikitext103_1024/base_data_128.txt'
    merged_sent, doc_labels = load_base_data(corpus_path, quiet=(worker_id != 0))
    datasets = split_into_chunk(list(range(len(merged_sent))), nworker)

    # vocab
    vocab_path = '/apdcephfs/share_916081/shared_info/ponybwcao/phrase_extraction/vocabulary/wiki103_merged.vocab'
    tokenizer = Tokenizer(vocab_path, mode='word_end_v2', lower_case=False, load_dict=True, tree=False, quiet=(worker_id != 0))

    # topk scores
    topk = 5
    ref_score_f = open(f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/retrieve_doc/output/wikitext103/copyisallyouneed/docs/doc_ref_8_split/scores/top{topk}_scores_{worker_id}.jsonl', 'r')

    # model
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/bak/wikitext103/copyisallyouneed/best_new_inbatch_prebatch256_indoc2_100000.pt')
    # agent.load_model(f'{args["root_dir"]}/ckpt/wikitext103/copyisallyouneed/train/{args["model_name"]}.pt')
    print(f'[!] init model over')

    output_path = f'/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_all_phrase_ref_re1/tokenization_result_{worker_id}.jsonl'

    chunk_size = 10000
    all_result = []
    ref_scores = []
    for doc_idx in tqdm.tqdm(datasets[worker_id], desc=f'Worker{worker_id} processing', disable=(worker_id != 0)):
        query = merged_sent[doc_idx]
        if len(ref_scores) == 0:
            ref_scores = load_lines_chunk(ref_score_f, chunk_size)
            ref_scores = [json.loads(x) for x in ref_scores]
        topk_phrase_map = ref_scores.pop(0)

        query_idx, merged_score_vocab, doc_ref_vocab = load_scores(topk_phrase_map)
        if query_idx != doc_idx:
            print('query id wrong!', doc_idx, query_idx)
            exit()
        max_score_vocab = {k: max(v) for k, v in merged_score_vocab.items()}
        global_vocab_logp = tokenizer.merge_word_freq_map(max_score_vocab, mode='relevance_score', position=True)
        references = choose_reference(query, global_vocab_logp, merged_score_vocab, doc_ref_vocab, doc_labels, merged_sent)
        
        if references:
            all_result.append({"results": references, "index": doc_labels[doc_idx]})

        if len(all_result) == chunk_size:
            with open(output_path, 'a') as f:
                for result in all_result:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            all_result = []

    ref_score_f.close()
    if len(all_result) > 0:
            with open(output_path, 'a') as f:
                for result in all_result:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            all_result = []

def compute_tok_phrase_rate():
    from transformers import AutoTokenizer
    tok_cnt = 0
    phrase_cnt = 0
    vocab = AutoTokenizer.from_pretrained('/apdcephfs/share_916081/shared_info/ponybwcao/PLM/GPT2-small')
    vocab.add_tokens(['<|endoftext|>'])
    for i in range(8):
        path = f'/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_all_phrase_ref_check_valid/tokenization_result_{i}.jsonl'
        with open(path) as f:
            for line in tqdm(f, desc=f'processing {i}', disable=True):
                data = json.loads(line.strip())['results']
                phrase, _ = data[0]
                toks = vocab(phrase, add_special_tokens=False)['input_ids']
                tok_cnt += len(toks)
                for phrase, metadata in data[1:]:
                    toks = vocab(phrase, add_special_tokens=False)['input_ids']
                    if metadata:
                        phrase_cnt += 1
                        tok_cnt += len(toks) - 1
                    else:
                        tok_cnt += len(toks)
        print(tok_cnt, phrase_cnt)
        total_num = tok_cnt + phrase_cnt
        print(tok_cnt / total_num, phrase_cnt / total_num)
    # old version
    # 20613561 76359991
    # 0.21256889713599436 0.7874311028640056

def compute_phrase_avg_len():
    base_data = {}
    with open(f'../data/wikitext103_1024/base_data_128.txt') as f:
        for line in tqdm(f.readlines()):
            line = line.strip().split('\t')
            chunk = ' '.join(line[:-1])
            id_label = line[-1].strip()
            if id_label:
                base_data[id_label] = chunk
    cnt = 0
    word_num = 0
    for i in range(8):
        path = f'/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_all_phrase_ref_check_valid/tokenization_result_{i}.jsonl'
        with open(path) as f:
            for line in tqdm(f, desc=f'processing {i}', disable=True):
                data = json.loads(line.strip())['results']
                for phrase, metadata in data:
                    if metadata:
                        doc_idx, st, end = metadata
                        doc_text = base_data[doc_idx]
                        phrase = doc_text[st: end]
                        word_num += len(phrase.split())
                    else:
                        word_num += 1
                    cnt += 1
        print(word_num, cnt, word_num / cnt)

def split_train_dev():
    data_dir = '/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_all_phrase_ref_check_valid'
    train_dir = f'{data_dir}/train'
    dev_dir = f'{data_dir}/dev'
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(dev_dir):
        os.mkdir(dev_dir)

    dev_num_pre_file = 150
    fns = os.listdir(data_dir)
    fns = [fn for fn in fns if fn.endswith('.jsonl')]
    for fn in tqdm(fns):
        input_path = os.path.join(data_dir, fn)
        train_path = os.path.join(train_dir, fn)
        dev_path = os.path.join(dev_dir, fn)
        with open(input_path) as f:
            data = f.readlines()
        
        dev_idx = random.sample(range(len(data)), dev_num_pre_file)
        dev_data = [data[idx] for idx in dev_idx]
        train_data = [
            item for index, item in enumerate(data)
            if not index in dev_idx
        ]
        with open(train_path, 'w') as fOut:
            for line in train_data:
                fOut.write(line)

        with open(dev_path, 'w') as fOut:
            for line in dev_data:
                fOut.write(line)

def split_train_dev_file(dev_num=1000):
    data_dir = '/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_all_phrase_ref_check_valid_merged'
    input_path = f'{data_dir}/merged.jsonl'

    with open(input_path) as f:
        data = f.readlines()
    
    dev_idx = random.sample(range(len(data)), dev_num)
    dev_data = [data[idx] for idx in dev_idx]
    train_data = [
        item for index, item in enumerate(data)
        if not index in dev_idx
    ]
    random.shuffle(train_data)
    train_data = split_into_chunk(train_data, 8)
    for i in range(8):
        train_path = f'{data_dir}/train/train_{i}.jsonl'
        with open(train_path, 'w') as fOut:
            for line in train_data[i]:
                fOut.write(line)

    dev_path = f'{data_dir}/dev/dev.jsonl'
    with open(dev_path, 'w') as fOut:
        for line in dev_data:
            fOut.write(line)

def preprocess_dev(**args):
    # vocab
    vocab_path = '/apdcephfs/share_916081/shared_info/ponybwcao/phrase_extraction/vocabulary/wiki103_merged.vocab'
    tokenizer = Tokenizer(vocab_path, mode='word_end_v2', lower_case=False, load_dict=False, tree=True, tree_mode='word')

    args['mode'] = 'test'

    config = load_config(args)
    args.update(config)
    agent = load_model(args)

    base_data = {}
    with open(f'../data/wikitext103_1024/base_data_128.txt') as f:
        for line in tqdm(f.readlines()):
            line = line.strip().split('\t')
            chunk = ' '.join(line[:-1])
            id_label = line[-1].strip()
            if id_label:
                base_data[id_label] = chunk

    dev_dir = '/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_all_phrase_ref_check_valid_merged/dev/'
    dev_path = f'{dev_dir}/dev.jsonl'
    gpt2_batch = []
    bert_batch = []
    phrase_doc_dict, phrase_global_idx_map_list = {}, []
    global_phrase_cnt = 0
    cache_phrases, cache_doc_idxs, cache_doc_toks = [], [], []
    tok_num = len(agent.model.tokenizer)
    with open(dev_path) as f:
        query_idx = 0
        for line in tqdm(f.readlines(), desc='Processing dev data'):
            line = line.strip()
            if not line:
                continue
            gpt_phrase_list = []
            line = json.loads(line)
            phrase_ref, index = line['results'], line['index']
            query_text = ' '.join([phrase for phrase, ref in phrase_ref])
            cur_pos = 0
            for phrase_idx, (phrase, ref) in enumerate(phrase_ref):
                if phrase_idx > 0:
                    phrase_ = ' ' + phrase
                else:
                    phrase_ = phrase
                phrase_tok = agent.model.tokenizer(phrase_, add_special_tokens=False)['input_ids']
                tok_labels = deepcopy(phrase_tok)
                phrase_st_pos = [(query_idx, -1, -1) for _ in range(len(phrase_tok))]
                st_pos_ = query_text.find(phrase, cur_pos)
                phrase_st_pos[0] = (query_idx, st_pos_, st_pos_ + len(phrase))
                cur_pos += len(phrase) + 1
                if not ref or phrase_idx == 0:
                    gpt_phrase_list.extend([(tok, tok, st_pos) for tok, st_pos in zip(tok_labels, phrase_st_pos)])
                    continue
                doc_idx, st, end = ref
                if not doc_idx in phrase_doc_dict:
                    cache_doc_idxs.append(doc_idx)
                    bert_phrase_list = []
                    phrase_doc_dict[doc_idx] = len(phrase_global_idx_map_list)
                    pos_map = {}

                    text = base_data[doc_idx]
                    phrase_list = tokenizer.search_candidates(text, phrase=True, return_list=True)
                    item = agent.model.bert_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                    cache_doc_toks.append([agent.model.bert_tokenizer.cls_token_id] + item['input_ids'] + [agent.model.bert_tokenizer.sep_token_id])
                    start_mapping = [s for s, e in item['offset_mapping']]
                    end_mapping = [e for s, e in item['offset_mapping']]
                    for doc_phrase, doc_st_pos in phrase_list:
                        doc_end_pos = doc_st_pos + len(doc_phrase)
                        try:
                            # +1 cause the [CLS] token is added
                            start_index = start_mapping.index(doc_st_pos) + 1
                            end_index = end_mapping.index(doc_end_pos) + 1
                        except:
                            continue
                        pos_map[(doc_st_pos, doc_end_pos)] = global_phrase_cnt
                        bert_phrase_list.append((start_index, end_index, doc_phrase))
                        # bert_phrase_list.append((doc_phrase, doc_st_pos, doc_end_pos, start_index, end_index))
                        global_phrase_cnt += 1
                    phrase_global_idx_map_list.append(pos_map)
                    cache_phrases.append(bert_phrase_list)
                doc_inbatch_idx = phrase_doc_dict[doc_idx]
                pos_map = phrase_global_idx_map_list[doc_inbatch_idx]
                if (st, end) in pos_map:
                    phrase_global_idx = pos_map[(st, end)]
                    tok_labels[0] = phrase_global_idx + tok_num
                gpt_phrase_list.extend(zip(phrase_tok, tok_labels, phrase_st_pos))
            if len(gpt_phrase_list) > 1024:
                print(len(gpt_phrase_list))
                exit()
            gpt2_batch.append({"index": index, "query": query_text, "results": gpt_phrase_list})
            query_idx += 1
        bert_batch = list(zip(cache_doc_idxs, cache_doc_toks, cache_phrases))
    with open(f'{dev_dir}/dev_query_tok.jsonl', 'w') as fOut:
        for x in gpt2_batch:
            fOut.write(json.dumps(x, ensure_ascii=False) + '\n')
    with open(f'{dev_dir}/dev_document_tok.jsonl', 'w') as fOut:
        for x in bert_batch:
            fOut.write(json.dumps(x, ensure_ascii=False) + '\n')

def test_dev(**args):

    args['mode'] = 'test'
    local_rank = args['local_rank']

    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'../ckpt/wikitext103/copyisallyouneed/train_pipeline/{args["model_name"]}.pt')
    
    all_result, tok_counter, phrase_counter = agent.model.evaluate(quiet=False)

    with open(f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/raw_files/dev_result/0607_{args["model_name"]}.log', 'w') as f:
        sorted_keys = list(all_result.keys())
        sorted_keys.sort()
        for k in sorted_keys:
            v = all_result[k]
            if 'token' in k:
                f.write('%s: %.4f\n' % (k, v / tok_counter))
            elif 'phrase' in k:
                f.write('%s: %.4f\n' % (k, v / phrase_counter))
            else:
                f.write('%s: %.4f\n' % (k, v / (phrase_counter + tok_counter)))

def gpt_test_dev(**args):

    args['mode'] = 'test'
    args['model'] = 'gpt2'
    local_rank = args['local_rank']

    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'../ckpt/wikitext103/gpt2/train/{args["model_name"]}.pt')
    
    acc = agent.model.evaluate(quiet=False)
    print(acc)

    
def analysis_emb(**args):

    args['mode'] = 'test'
    local_rank = args['local_rank']

    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'../ckpt/wikitext103/copyisallyouneed/train_pipeline/{args["model_name"]}.pt')
    
    phrase_num, mean, std = agent.model.analysis_emb(quiet=False)
    mean = mean.tolist()
    std = std.tolist()
    print(phrase_num)
    with open('emb_mean_std.jsonl', 'w') as f:
        f.write(json.dumps({'mean': mean, 'std': std}, ensure_ascii=False) + '\n')

def test_EM():
    base_data = {}
    with open('/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/data/wikitext103_1024/base_data_128.txt') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split('\t')
            chunk = ' '.join(line[:-1])
            id_label = line[-1].strip()
            if id_label:
                base_data[id_label] = chunk
    with open('/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_all_phrase_ref_check_valid_merged/train_EM1/tmp.jsonl') as f:
        for line in f:
            item = json.loads(line)
            phrase_ref = item['results']
            phrases = [x[0] for x in phrase_ref]
            refs = [x[1] for x in phrase_ref]
            for i, (phrase, ref) in enumerate(zip(phrases, refs)):
                ref_phrase = ''
                if ref:
                    suffix = ' '.join(phrases[i:])
                    doc_idx, st, end = ref
                    ref_phrase = base_data[doc_idx][st: end]
                    if not re.match(r'%s\b' % re.escape(ref_phrase), suffix):
                        print('[!]', suffix, '**', phrase, '**', ref_phrase, "**")
                        exit()
                print(phrase, ref_phrase)
            break

def precompute_step(**args):
    all_batch_cnt = []
    for worker_id in tqdm(range(8)):
        # path = f'/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_all_phrase_ref_check_valid_merged/train/train_{worker_id}.jsonl'
        path = f'/apdcephfs/share_916081/shared_info/ponybwcao/phrase_extraction/data/wikipedia/references/cluster{args["cluster_idx"]}_training{worker_id}.jsonl'
        batch_cnt = 0
        doc_set = set()
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                phrase_ref = data['results']
                for phrase, ref in phrase_ref:
                    if ref:
                        doc_set.add(ref[0])
                        if len(doc_set) == 32:
                            batch_cnt += 1
                            doc_set = set()

        all_batch_cnt.append(batch_cnt)
    print(all_batch_cnt)
        

if __name__ == "__main__":
    args = vars(parser_args())
    if args['mode'] == 'encode':
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args['worker_idx'] = args['local_rank']
        encode_phrase(**args)
    elif args['mode'] == 'generation_test':
        result = generation_test(**args)
        with open(f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/raw_files/random_runs_en_wiki_testset/{args["dataset"]}_copy_phrase_test_{args["decoding_method"]}_{args["model_name"]}.json', 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    elif args['mode'] == 'generation':
        result = generation(**args)
        # with open(f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/raw_files/random_runs_en_wiki_testset/{args["dataset"]}_copy_queryside_{args["decoding_method"]}_best_prebatch_neg0_pretrain40w_100w_20000.json', 'w') as f:
        #     json.dump(result, f, indent=4, ensure_ascii=False)
    elif args['mode'] == 'generation_gt':
        result = generation_from_gt_docs(**args)
        with open(f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/raw_files/random_runs_en_wiki_testset/{args["dataset"]}_copy_queryside_{args["decoding_method"]}_best_prebatch_neg0_pretrain40w_100w_20000.json', 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    elif args["mode"] == 'merge':
        merge_result()
    elif args["mode"] == 'merge_cluster':
        merge_cluster_result()
    elif args["mode"] == 'cluster':
        # clustering_phrase()
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        clustering_phrase_multiproc(**args)
    elif args["mode"] == 'test':
        # test()
        # compute_tok_phrase_rate()
        # compute_phrase_avg_len()
        test_dev(**args)
        # analysis_emb(**args)
        # test_EM()
        # gpt_test_dev(**args)
    elif args["mode"] == 'pq':
        train_quantizer()
    elif args["mode"] == 'index':
        if args['index_type'] == 'opq':
            train_index_opq()
        else:
            train_index(args['index_type'])
    elif args["mode"] == 'generate_from_candidate':
        generation_from_candidate_docs(**args)    
    elif args['mode'] == 'dev_data':
        # split_train_dev()
        split_train_dev_file(dev_num=1000)
        preprocess_dev(**args)
    elif args['mode'] == 'EM':
        regenerate_ref_retrieve(**args)
    elif args['mode'] == 'step':
        precompute_step(**args)
    elif args["mode"] == 'obqa':
        OBQA_test(**args)
    elif args["mode"] == 'obqa_gpt':
        OBQA_test_gpt2(**args)
    else:
        exit('Not implemented.')