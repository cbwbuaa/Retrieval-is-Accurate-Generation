from header import *
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataloader import *
from models import *
from config import *
from build_phrase_index import build_index
from time import time
import faiss.contrib.torch_utils
import sys
import json
sys.path.append('../data/')

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model_size', type=str, default='small')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--search_topk', type=int, default=128)
    parser.add_argument('--prefix_length', type=int, default=128)
    parser.add_argument('--max_gen_len', type=int, default=128)
    parser.add_argument('--lambda', type=float, default=0.118)
    parser.add_argument('--alpha', type=float, default=0.00785)
    parser.add_argument('--decoding_method', type=str, default='nucleus_sampling')
    parser.add_argument('--mode', type=str, default='generation')
    return parser.parse_args()

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-np.inf):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits

class KNNLM(nn.Module):

    def __init__(self, **args):
        super(KNNLM, self).__init__()
        self.args = args
        self.tokenizer = GPT2Tokenizer.from_pretrained(f'/apdcephfs/share_916081/ponybwcao/PLM/GPT2-{args["model_size"]}')
        self.model = GPT2LMHeadModel.from_pretrained(f'/apdcephfs/share_916081/ponybwcao/PLM/GPT2-{args["model_size"]}')
        self.model = self.model.cuda()
        self.model.eval()

        self.pad = self.tokenizer.eos_token_id
        self.unk = self.tokenizer.unk_token_id
        self.special_tokens = set([self.pad])
        self.gen_loss_fct = nn.NLLLoss(ignore_index=self.tokenizer.eos_token_id, reduction='none')

        embeddings = np.load('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/knn_index/embedding_tok/small_sampled/emb.npy')
        self.next_tokens = np.load('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/knn_index/embedding_tok/small_sampled/tok.npy')
        self.index = build_index(gpu_list=[1, 2, 3], dim=768)
        self.index.add(embeddings)

    @torch.no_grad()
    def calculate_ppl(self, text_list):
        self.model.eval()
        ids = self.tokenizer(text_list, return_attention_mask=False)['input_ids']
        all_length = [len(x) for x in ids]
        all_gpt_ids = pad_sequence([torch.LongTensor(x) for x in ids], padding_value=self.tokenizer.eos_token_id, batch_first=True).cuda()
        all_gpt2_mask = pad_sequence([torch.ones(len(x)).to(torch.long) for x in ids], padding_value=0, batch_first=True).cuda()
        output = self.model(input_ids=all_gpt_ids, attention_mask=all_gpt2_mask, output_hidden_states=True)
        hidden = output['hidden_states'][-1] # (B, S, H)

        query = hidden[:, :-1].reshape(-1, hidden.shape[-1])
        query_mask = all_gpt2_mask[:, 1:].reshape(-1).to(torch.bool)
        query = query[query_mask]
        label = all_gpt_ids[:, 1:].reshape(-1)
        label = label[query_mask]
        LM_logits = output.logits[:, :-1].reshape(-1, output.logits.shape[-1])
        LM_logits = LM_logits[query_mask]
        
        window_size = 256
        losses = []
        for tok_st_pos in range(0, len(query), window_size):
            tok_end_pos = min(tok_st_pos + window_size, len(query))
            sub_query = query[tok_st_pos: tok_end_pos]
            sub_LM_logits = LM_logits[tok_st_pos: tok_end_pos]
            sub_label = label[tok_st_pos: tok_end_pos]

            dists, cands = self.index.search(sub_query.cpu().numpy(), self.args['search_topk'])
            cands = torch.LongTensor([[int(self.next_tokens[i]) for i in j] for j in cands]).unsqueeze(-1) # [S, K, 1]
            dists = torch.from_numpy(dists).cuda()    # [S, K]
            dists = F.softmax(-self.args['alpha'] * dists, dim=-1).unsqueeze(-1).cpu()   # [S, K, 1]
            knn_logits = torch.zeros(len(sub_query), self.args['search_topk'], len(self.tokenizer))   # [S, K, V]
            knn_logits.scatter_(2, cands, dists)
            knn_logits = knn_logits.sum(dim=1).cuda()    # [S, V]
            new_logits = self.args['lambda'] * knn_logits + (1 - self.args['lambda']) * F.softmax(sub_LM_logits, dim=-1)    # [S, V]
            new_logits = new_logits.log()
            loss = self.gen_loss_fct(new_logits.view(-1, new_logits.size(-1)), sub_label.view(-1))
            losses.append(loss)
        loss = torch.hstack(losses)
        ppl = []
        cur_idx = 0
        for instance_idx, length in enumerate(all_length):
            st_idx = cur_idx
            end_idx = st_idx + length - 1
            ppl.append(math.exp(loss[st_idx: end_idx].mean().item()))
            cur_idx = end_idx
        return ppl

    @torch.no_grad()
    def _calculate_ppl(self, text_list):
        self.model.eval()
        ids = self.tokenizer(text_list, return_attention_mask=False)['input_ids']
        all_length = [len(x) for x in ids]
        all_gpt_ids = pad_sequence([torch.LongTensor(x) for x in ids], padding_value=self.tokenizer.eos_token_id, batch_first=True).cuda()
        all_gpt2_mask = pad_sequence([torch.ones(len(x)).to(torch.long) for x in ids], padding_value=0, batch_first=True).cuda()
        output = self.model(input_ids=all_gpt_ids, attention_mask=all_gpt2_mask, output_hidden_states=True)
        hidden = output['hidden_states'][-1] # (B, S, H)

        query = hidden[:, :-1].reshape(-1, hidden.shape[-1])
        query_mask = all_gpt2_mask[:, 1:].reshape(-1).to(torch.bool)
        query = query[query_mask]
        label = all_gpt_ids[:, 1:].reshape(-1)
        label = label[query_mask]
        LM_logits = output.logits[:, :-1].reshape(-1, output.logits.shape[-1])
        LM_logits = LM_logits[query_mask]
        
        window_size = 64
        losses = []
        for tok_st_pos in range(0, len(query), window_size):
            tok_end_pos = min(tok_st_pos + window_size, len(query))
            sub_query = query[tok_st_pos: tok_end_pos]
            sub_LM_logits = LM_logits[tok_st_pos: tok_end_pos]
            sub_label = label[tok_st_pos: tok_end_pos]

            dists, cands = self.index.search(sub_query.cpu().numpy(), self.args['search_topk'])
            cands = torch.LongTensor([[int(self.next_tokens[i]) for i in j] for j in cands]).unsqueeze(-1).cuda() # [S, K, 1]
            dists = torch.from_numpy(dists).cuda()    # [S, K]
            dists = F.softmax(-self.args['alpha'] * dists, dim=-1).unsqueeze(-1)   # [S, K, 1]
            knn_logits = torch.zeros(len(sub_query), self.args['search_topk'], len(self.tokenizer)).cuda()   # [S, K, V]
            knn_logits.scatter_(2, cands, dists)
            knn_logits = knn_logits.sum(dim=1)    # [S, V]
            new_logits = self.args['lambda'] * knn_logits + (1 - self.args['lambda']) * F.softmax(sub_LM_logits, dim=-1)    # [S, V]
            new_logits = new_logits.log()
            loss = self.gen_loss_fct(new_logits.view(-1, new_logits.size(-1)), sub_label.view(-1))
            losses.append(loss)
        loss = torch.hstack(losses)
        ppl = []
        cur_idx = 0
        for instance_idx, length in enumerate(all_length):
            st_idx = cur_idx
            end_idx = st_idx + length - 1
            ppl.append(math.exp(loss[st_idx: end_idx].mean().item()))
            cur_idx = end_idx
        return ppl

    @torch.no_grad()
    def generate_new_logits(self, logits, hidden, topk=10, temp=100, ids=None):
        dists, cands = self.index.search(hidden.unsqueeze(0).cpu().numpy(), topk)
        cands = torch.LongTensor([int(self.next_tokens[i]) for i in cands[0]]).cuda()
        dists = torch.tensor(dists[0]).cuda()
        knn_logits = torch.zeros(topk, len(self.tokenizer)).cuda()    # [K, V]
        knn_logits[range(topk), cands] = F.softmax(-self.args['alpha']*dists, dim=-1)
        knn_logits = knn_logits.sum(dim=0)    # [V]
        new_logits = self.args['lambda'] * knn_logits + (1 - self.args['lambda']) * F.softmax(logits, dim=-1)
        return new_logits
    
    @torch.no_grad()
    def generate_new_logits_batch(self, logits, hidden, topk=10, temp=100, ids=None):
        dists, cands = self.index.search(hidden.cpu().numpy(), topk)
        cands = torch.LongTensor([[int(self.next_tokens[i]) for i in x] for x in cands]).cuda()
        dists = torch.tensor(dists).cuda() # B, K
        knn_logits = torch.zeros(len(logits), topk, len(self.tokenizer)).cuda()    # [B, K, V]
        knn_probs = F.softmax(-self.args['alpha']*dists, dim=-1) # B, K
        for instance_idx in range(len(logits)):
            knn_logits[instance_idx, range(topk), cands[instance_idx]] = knn_probs[instance_idx]
        knn_logits = knn_logits.sum(dim=1)    # [B, V]
        new_logits = self.args['lambda'] * knn_logits + (1 - self.args['lambda']) * F.softmax(logits, dim=-1)
        return new_logits

    @torch.no_grad()
    def greedy_search(self, ids, max_length):
        generated = []
        past_key_values = None
        for _ in range(max_length):
            output = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = output['past_key_values']
            hidden = output['hidden_states'][-1][-1, -1, :]
            next_token_logits = output['logits'][-1, -1, :]
            next_token_logits = self.generate_new_logits(next_token_logits, hidden, topk=self.args['search_topk'], ids=ids)
            next_token_logits[self.unk] = -np.inf
            ids = torch.argmax(next_token_logits, dim=-1).reshape(1, 1)
            generated.append(ids.item())
        string = self.tokenizer.decode(generated)
        return string

    @torch.no_grad()
    def nucleus_sampling(self, ids, max_length, top_p, get_time_cost=False):
        generated = []
        past_key_values = None
        for _ in range(max_length):
            output = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = output['past_key_values']
            hidden = output['hidden_states'][-1][-1, -1, :]
            next_token_logits = output['logits'][-1, -1, :]
            next_token_logits = self.generate_new_logits(next_token_logits, hidden, topk=self.args['search_topk'], ids=ids)
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=0, top_p=top_p, filter_value=0)
            filtered_logits[self.unk] = 0
            # no softmax for nucleus sampling
            ids = torch.multinomial(filtered_logits, num_samples=1).reshape(1, 1)
            generated.append(ids.item())
        string = self.tokenizer.decode(generated)
        return string

    @torch.no_grad()
    def nucleus_sampling_batch(self, ids, max_length, top_p, get_time_cost=False):
        generated = [[] for _ in range(len(ids))]
        for _ in range(max_length):
            output = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True
            )
            hidden = output['hidden_states'][-1][:, -1, :]
            next_token_logits = output['logits'][:, -1, :]
            next_token_logits = self.generate_new_logits_batch(next_token_logits, hidden, topk=self.args['search_topk'], ids=ids)
            new_ids = []
            for instance_idx in range(len(ids)):
                filtered_logits = top_k_top_p_filtering(next_token_logits[instance_idx], top_k=0, top_p=top_p, filter_value=0)
                filtered_logits[self.unk] = 0
                # no softmax for nucleus sampling
                new_id = torch.multinomial(filtered_logits, num_samples=1)
                generated[instance_idx].append(new_id.item())
                new_ids.append(new_id)
            new_ids = torch.vstack(new_ids)
            ids = torch.hstack([ids, new_ids])
        string = [self.tokenizer.decode(x) for x in generated]
        return string

    @torch.no_grad()
    def forward(self, batch):
        self.model.eval()
        ids, ids_mask, vl = batch['ids'], batch['ids_mask'], batch['vl']
        output = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
        collection_rep, collection_target = [], []
        ids = ids.tolist()
        for rep, ids_, l in zip(output, ids, vl):
            collection_rep.append(rep[:l-1, :])
            collection_target.extend(ids_[1:l])
        collection_rep = torch.cat(collection_rep).cpu()
        assert len(collection_rep) == len(collection_target)
        collection_target = [str(i) for i in collection_target]
        return collection_rep, collection_target

    @torch.no_grad()
    def generate(self, prefix, decoding_method='nucleus_sampling', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False):
        # maximum 128 tokens
        input_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        input_ids = torch.LongTensor(input_ids).unsqueeze(dim=0).cuda()
        length = len(input_ids[0])
        bt = time()
        if decoding_method == 'nucleus_sampling':
            string = self.nucleus_sampling(
                input_ids,
                max_length=128,
                top_p=top_p,
            )
        elif decoding_method == 'greedy':
            string = self.greedy_search(
                input_ids,
                max_length=128,
            )
        return string, time() - bt

    @torch.no_grad()
    def generate_batch(self, prefix, generate_length=128, decoding_method='nucleus_sampling', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False):
        # make sure prefixes have the same token num
        encoded_dict = self.tokenizer(prefix, padding=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids'].cuda()
        bt = time()
        if decoding_method == 'nucleus_sampling':
            string = self.nucleus_sampling_batch(
                input_ids,
                max_length=generate_length,
                top_p=top_p,
                get_time_cost=get_time_cost
            )
        elif decoding_method == 'greedy':
            # string = self.greedy_search(
            #     input_ids,
            #     max_length=128,
            #     get_time_cost=get_time_cost
            # )
            pass
        return string, time() - bt

@torch.no_grad()
def main_generation(**args):
    kNN_LM = KNNLM(**args)
    kNN_LM.tokenizer.pad_token = kNN_LM.tokenizer.eos_token
    print(f'[!] init model over, knnlm with lambda {args["lambda"]}')

    prefix_length = args['prefix_length']
    generate_length = args['max_gen_len']
    test_num = -1
    batch_size = 16
    collection = []
    with open(f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/minipile/raw/test.jsonl') as f:
        # collect the valid prefixes
        texts = []
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            ids = kNN_LM.tokenizer.encode(line['text'], add_special_tokens=False)
            prefix, reference = ids[:prefix_length], ids[prefix_length:]
            if len(prefix) == prefix_length and len(reference) >= generate_length:
                prefix = kNN_LM.tokenizer.decode(prefix)
                reference = kNN_LM.tokenizer.decode(reference[:generate_length])
                texts.append((prefix, reference))
                if len(texts) == test_num:
                    break
        print(f'[!] collect {len(texts)} valid samples which have at least {prefix_length} tokens in prefix')

    # texts = texts[:test_num]
    collection = []  
    for st in tqdm(range(0, len(texts), batch_size)):
        end = min(st + batch_size, len(texts))
        batch_texts = texts[st: end]
        batch_prefix = [x[0] for x in batch_texts]
        batch_reference = [x[1] for x in batch_texts]

        batch_output, time_cost = kNN_LM.generate_batch(batch_prefix, generate_length=generate_length, decoding_method=args['decoding_method'], top_k=-1, top_p=0.95, temp=1., get_time_cost=True)
        for prefix, reference, output in zip(batch_prefix, batch_reference, batch_output):
            collection.append({
                'prefix': prefix, 
                'reference': reference, 
                'text': output,
                'time_cost': time_cost / len(batch_texts)
            })
    return collection

@torch.no_grad()
def MCQA(**args):
    from QA_test import load_QA_dataset
    kNN_LM = KNNLM(**args)
    print(f'[!] init model over, knnlm with lambda {args["lambda"]}')

    dataset_name = args['dataset']
    dataset = load_QA_dataset(dataset_name)
    print(f'load {len(dataset)} examples from {dataset_name}')
    result = []
    for item in tqdm(dataset, desc=dataset_name):
        question = item['question']
        choices = item['choices']
        gt = item['label']
        text_list = [question + ' ' + x for x in choices]
        all_score = kNN_LM.calculate_ppl(text_list)
        pred = int(np.argmin(all_score))
        result.append(pred == gt)
    print(sum(result) / len(result))
    print('*' * 30)

if __name__ == "__main__":
    args = vars(parser_args())
    mode = args['mode']
    if mode == 'generation':
        result = main_generation(**args)
        with open(f'/apdcephfs/share_916081/ponybwcao/tmp/copyisallyouneed_v2/log/generation/kNNLM_{args["prefix_length"]}_{args["max_gen_len"]}_{args["search_topk"]}_{args["decoding_method"]}.json', 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    elif mode == 'QA':
        MCQA(**args)
