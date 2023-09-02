from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import torch
import time
import pickle
from tqdm import tqdm
import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence

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
        chunk_size = 2000000
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
                            print(f'Dump into chunk{inner_idx}')
                            np.save(f"{args['output_dir']}/{args['model_size']}/emb_{args['worker_idx']}_{inner_idx}.npy", np.vstack(embeddings))
                            np.save(f"{args['output_dir']}/{args['model_size']}/tok_{args['worker_idx']}_{inner_idx}.npy", np.array(tokens, dtype=np.int32))
                            inner_idx += 1
                            embeddings = []
                            tokens = []
            if len(tokens) > 0:
                if args['local_rank'] == 0:
                    print(f'Dump into chunk{inner_idx}')
                np.save(f"{args['output_dir']}/{args['model_size']}/emb_{args['worker_idx']}_{inner_idx}.npy", np.vstack(embeddings))
                np.save(f"{args['output_dir']}/{args['model_size']}/tok_{args['worker_idx']}_{inner_idx}.npy", np.array(tokens, dtype=np.int32))

    #Load GPT2
    tokenizer = GPT2Tokenizer.from_pretrained(f'/apdcephfs/share_916081/ponybwcao/PLM/GPT2-{args["model_size"]}')
    model = GPT2LMHeadModel.from_pretrained(f'/apdcephfs/share_916081/ponybwcao/PLM/GPT2-{args["model_size"]}', return_dict=True)
    model = model.cuda()
    model = model.eval()
    
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

if __name__ == '__main__':
    args = vars(parser_args())
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    args['worker_idx'] = args['local_rank']
    gpt_encoding(args)