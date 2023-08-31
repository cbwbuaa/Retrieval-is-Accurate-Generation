from header import *
import sys, random
import numpy as np
from time import time
from tqdm import tqdm
import collections
from .util_func import *
import faiss
import faiss.contrib.torch_utils
from copy import deepcopy
# torch.set_printoptions(profile='full')

class CopyisallyouneedPrefixOnly(nn.Module):

    def __init__(self, **args):
        super(CopyisallyouneedPrefixOnly, self).__init__()
        self.args = args
        if args['random_initialize']:
            if args['local_rank'] == 0:
                print('[!] model random initialized.')
            # GPT model
            GPT_config = AutoConfig.from_pretrained(
                self.args['prefix_encoder_model'][self.args['lang']]
            )
            self.model = GPT2LMHeadModel(GPT_config)
            # self.model = AutoModel.from_config(GPT_config)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(self.args['prefix_encoder_model'][self.args['lang']])
        # GPT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args['prefix_encoder_tokenizer'][self.args['lang']])

        self.vocab_size = len(self.tokenizer)
        self.pad = self.tokenizer.pad_token_id if self.args['lang'] == 'zh' else self.tokenizer.eos_token_id
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        assert self.token_embeddings.shape[0] == self.vocab_size
        
        self.phrase_dim = self.args['phrase_dim'] if 'phrase_dim' in self.args else self.model.config.hidden_size
        if self.phrase_dim != self.model.config.hidden_size:
            self.dim_proj = nn.Linear(self.model.config.hidden_size, self.phrase_dim)
        else:
            self.dim_proj = None
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return output

    def get_token_loss(self, ids, hs, AR_mask, token_reps, do_eval=True):
        AR_mask = AR_mask.to(torch.bool)
        label = ids[:, 1:].reshape(-1)
        label_mask = AR_mask[:, 1:].reshape(-1)
        label = label[label_mask]
        logits = torch.matmul(
            hs[:, :-1, :],
            token_reps.t()
        )
        logits = logits.reshape(-1, logits.shape[-1])
        logits = logits[label_mask]
        # logits /= self.args['temp']
        loss = self.gen_loss_fct(logits, label)
        if not do_eval:
            return loss
        else:
            chosen_tokens = torch.max(logits, dim=-1)[1]
            # print(self.tokenizer.decode(chosen_tokens[:100]))
            # print('-'*20)
            # print(self.tokenizer.decode(label[:100]))
            gen_acc = (chosen_tokens == label).to(torch.long)
            gen_acc = gen_acc.sum().item() / len(gen_acc)
            return loss, gen_acc

    def forward(self, batch):  
        model_st_time = time()
        token_reps = self.token_embeddings
        if self.dim_proj is not None:
            token_reps = self.dim_proj(token_reps)
        
        ## gpt2 query encoder
        ids, ids_mask, AR_mask = batch['gpt2_ids'].cuda(), batch['gpt2_mask'].cuda(), batch['AR_mask'].to(torch.bool).cuda()
        query_hidden_states = \
        self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True).hidden_states[-1]
        if self.dim_proj is not None:
            query_hidden_states = self.dim_proj(query_hidden_states)
        AR_loss, acc = self.get_token_loss(ids, query_hidden_states, AR_mask, token_reps, do_eval=True)
        # print('AR_loss_acc:', AR_loss, acc)

        phrase_rep = batch['phrase_embedding']
        if phrase_rep is None:
            return AR_loss, {}
        
        phrase_rep = phrase_rep.cuda()
        candidate_reps = torch.cat([
            token_reps,
            phrase_rep], dim=0)
    
        false_neg_mask = batch['false_neg_mask'].cuda()
        false_neg_mask = false_neg_mask.reshape(-1, false_neg_mask.shape[-1])
        labels = batch['phrase_label'].cuda()
        
        query = query_hidden_states[:, :-1].reshape(-1, query_hidden_states.shape[-1])
        query_mask = AR_mask[:, 1:].reshape(-1).to(torch.bool)
        query = query[query_mask]
        labels = labels[:, :-1].reshape(-1)
        labels = labels[query_mask]
        phrase_indexes = labels >= self.vocab_size
        # print('query:', query.shape, labels.shape, phrase_indexes.sum().item())
        # token_indexes = ~phrase_indexes
        # phrase_num = phrase_indexes.sum()
        # token_num = token_indexes.sum()

        logits = torch.matmul(query, candidate_reps.t())
        logits /= self.args['temp']
        full_false_neg_mask = torch.ones_like(logits).to(torch.long)
        full_false_neg_mask[phrase_indexes] = false_neg_mask

        # print(false_neg_mask.shape, logits.shape)
        logits = torch.where(full_false_neg_mask.to(torch.bool), logits, torch.tensor(-np.inf).to(torch.half).cuda())
        if torch.isnan(logits).any():
            sys.stderr.write(f"logits nan\n")
            sys.stderr.write(f"logits: {logits}\n")
            sys.stderr.write(f"query: {self.tokenizer.decode(ids.reshape(-1))}\n")
            sys.stderr.write(f"doc: {self.bert_tokenizer.decode(dids.reshape(-1))}\n")
            sys.stderr.write(f"labels: {labels}\n")
            exit()
        # ## split the token accuaracy and phrase accuracy
        # phrase_acc = logits[phrase_indexes].max(dim=-1)[1] == labels[phrase_indexes]
        # phrase_acc = phrase_acc.to(torch.float).mean().item()
        # if token_num > 0: # no token
        #     token_acc = logits[token_indexes].max(dim=-1)[1] == labels[token_indexes]
        #     token_acc = token_acc.to(torch.float).mean().item()
        # else:
        #     token_acc = -1
        if self.args['loss_type'] == 'focal_loss':
            logit_logsoftmax = F.log_softmax(logits, dim=-1)
            logit_softmax = torch.exp(logit_logsoftmax)
            p = logit_softmax[range(len(logits)), labels]
            logp = logit_logsoftmax[range(len(logits)), labels]
            alpha = 0.25
            gamma = 2
            loss_ = -alpha * (1 - p) ** gamma * logp
            loss = loss_.mean()
            # beta = self.args['beta']
            # loss = 0
            # if phrase_num > 0:
            #     loss += beta * loss_[phrase_indexes].mean()
            # if token_num > 0:
            #     loss += (1 - beta) * loss_[token_indexes].mean()
        else:
            loss = F.log_softmax(logits, dim=-1)
            loss = loss[range(len(logits)), labels]
            loss = (-loss.sum(dim=-1)).mean()
        loss += AR_loss
        if torch.isnan(loss).any():
            sys.stderr.write(f"loss nan\n")
            sys.stderr.write(f"loss: {loss}\n")
            sys.stderr.write(f"logits: {logits}\n")
            sys.stderr.write(f"query: {self.tokenizer.decode(ids.reshape(-1))}\n")
            sys.stderr.write(f"doc: {self.bert_tokenizer.decode(dids.reshape(-1))}\n")
            sys.stderr.write(f"labels: {labels}\n")
            exit()
        result_dict = {}
        # result_dict = {
        #     'phrase_acc': phrase_acc,
        #     'token_acc': token_acc
        # }
        model_end_time = time()
        # print("model time:", model_end_time - model_st_time)
        return (
            loss,  # phrase(tok) classification loss
            result_dict
        )

    @torch.no_grad()
    def evaluate(self, quiet=True):
        self.model.eval()
        phrase_embedding = np.memmap('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/phrase_embedding_index/PCA_emb_merged.npy', dtype=np.float32, mode='r', shape=(138543105, 128))

        data_fn = f'{self.args["training_data_dir"]}/validation_tok'
        with open(data_fn) as f:
            dev_data_queue = f.readlines()
        # 966 (1024)
        dev_data_queue = [json.loads(x) for x in tqdm(dev_data_queue)]
        dev_data_queue.sort(key=lambda x: len(x[0]), reverse=True)

        all_gpt_ids, all_valid_phrases, all_AR_mask = [], [], []
        for gpt_ids, valid_phrases, AR_mask in dev_data_queue:
            all_gpt_ids.append(gpt_ids)
            all_valid_phrases.append(valid_phrases)
            all_AR_mask.append(AR_mask)

        embeddings = []
        phrase_list = []
        emb_idx_map = {}
        emb_idx_list = []
        suffix_list = []
        for instance_idx, valid_phrases in enumerate(all_valid_phrases):
            suffix_list_ = ['' for _ in range(len(all_gpt_ids[instance_idx]))]
            for phrase, start_idx, end_idx, ref_pos, suffix in valid_phrases:
                emb_idx_map[(instance_idx, phrase, start_idx, end_idx)] = len(emb_idx_list)
                emb_idx_list.append(ref_pos)
                phrase_list.append(phrase)
                suffix_list_[start_idx] = suffix
            suffix_list.append(suffix_list_)
        if emb_idx_list:
            embeddings = torch.from_numpy(phrase_embedding[emb_idx_list])
        else:
            embeddings = None
        # print(embeddings.shape) # torch.Size([19806, 128])
        
        batch_size = 8
        topk = 20
        embeddings = embeddings.cuda()
        tok_reps = self.token_embeddings
        if self.dim_proj is not None:
            tok_reps = self.dim_proj(tok_reps)
        candidate_reps = torch.vstack([tok_reps, embeddings])
        # print(candidate_reps.shape) # torch.Size([70063, 128])

        for batch_st in tqdm(range(0, len(all_gpt_ids), batch_size)):
            batch_end = min(batch_st + batch_size, len(all_gpt_ids))
            ids_cpu = all_gpt_ids[batch_st: batch_end]
            AR_mask = all_AR_mask[batch_st: batch_end]
            batch_valid_phrases = all_valid_phrases[batch_st: batch_end]

            ids_cpu = pad_sequence([torch.LongTensor(x) for x in ids_cpu], padding_value=self.tokenizer.eos_token_id, batch_first=True)
            AR_mask = pad_sequence([torch.LongTensor(x) for x in AR_mask], padding_value=self.tokenizer.eos_token_id, batch_first=True)
            gpt2_mask = generate_mask(ids_cpu, pad_token_idx=self.tokenizer.eos_token_id)
            phrase_label = deepcopy(ids_cpu)
            seq_len = ids_cpu.shape[1]
            for local_idx, valid_phrases in enumerate(batch_valid_phrases):
                instance_idx = local_idx + batch_st
                for phrase, start_idx, end_idx, ref_pos, suffix in valid_phrases:
                    emb_idx = emb_idx_map[(instance_idx, phrase, start_idx, end_idx)]
                    phrase_label[local_idx][start_idx] = len(self.tokenizer) + emb_idx
            ids = ids_cpu.cuda()
            AR_mask = AR_mask.cuda()
            gpt2_mask = gpt2_mask.cuda()
            gpt_reps = \
            self.model(input_ids=ids, attention_mask=gpt2_mask, output_hidden_states=True).hidden_states[-1]
            if self.dim_proj is not None:
                gpt_reps = self.dim_proj(gpt_reps)
            AR_loss, acc = self.get_token_loss(ids, gpt_reps, AR_mask, tok_reps, do_eval=True)
            logits_ = torch.matmul(gpt_reps, candidate_reps.t())
            topk_logits, topk_indexs = torch.topk(logits_, k=topk, dim=-1)
            topk_indexs = topk_indexs.cpu()
            tok_counter, phrase_counter = 0, 0
            all_result = collections.defaultdict(int)
            k_list = [0, 1, 3, 5, 10, 20, 100]
            for instance_idx in range(len(topk_indexs)):
                tokens, topk_pred, label, suffix = ids_cpu[instance_idx][1:].tolist(), topk_indexs[instance_idx][:-1].tolist(), phrase_label[instance_idx][1:].tolist(), suffix_list[instance_idx][1:]
                for tok_idx, (tok_, topk_pred_, label_, suffix_) in enumerate(zip(tokens, topk_pred, label, suffix)):
                    # print(tok_, topk_pred_, label_, suffix_, self.tokenizer.decode(tok_))
                    if label_ < self.vocab_size: # token
                        tok_counter += 1
                        for k_idx in range(len(k_list) - 1):
                            st, end = k_list[k_idx], k_list[k_idx + 1]
                            if label_ in topk_pred_[st: end]:
                                for later_end in range(k_idx + 1, len(k_list)):
                                    all_result[f'token_hit@{k_list[later_end]}'] += 1
                                    all_result[f'global_acc@{k_list[later_end]}'] += 1
                                break
                    else:
                        phrase_counter += 1
                        # hit
                        for k_idx in range(len(k_list) - 1):
                            st, end = k_list[k_idx], k_list[k_idx + 1]
                            if label_ in topk_pred_[st: end]:
                                for later_end in range(k_idx + 1, len(k_list)):
                                    all_result[f'phrase_hit@{k_list[later_end]}'] += 1
                                break
                        if not suffix:
                            suffix_ = self.tokenizer.decode(tokens[tok_idx:tok_idx+72]) + ' '
                        else:
                            suffix_ += ' '
                        # print(suffix_)

                        # target_phrase = phrase_list[label_ - self.vocab_size]
                        # # print('target_phrase:', target_phrase)

                        # recall & acc
                        valid_counter = 0
                        for idx in range(topk):
                            pred_idx_ = topk_pred_[idx]
                            if (pred_idx_ >= self.vocab_size and suffix_.startswith(phrase_list[pred_idx_ - self.vocab_size] + ' ')) \
                                or (pred_idx_ < self.vocab_size and pred_idx_ == tok_):
                                # pred_phrase_ = phrase_list[pred_idx_ - self.vocab_size]
                                # print('pred_phrase_:', pred_phrase_)
                                valid_counter += 1
                                if valid_counter == 1:
                                    for tmp in range(idx, topk):
                                        if tmp + 1 in k_list[1:]: 
                                            all_result[f'phrase_acc@{tmp + 1}'] += 1
                                            all_result[f'global_acc@{tmp + 1}'] += 1
                            # Recall     
                            # if idx + 1 in k_list[1:]:
                            #     all_result[f'phrase_recall@{idx + 1}'] += valid_counter
            # for k, v in all_result.items():
            #     print('%s: %.4f' % (k, v / counter))
        return all_result, tok_counter, phrase_counter

    @torch.no_grad()
    def analysis_emb(self, quiet=True):
        data_dir = f'{self.args["training_data_dir"]}/dev'
        bert_batch, bert_indexs = [], []
        with open(f'{data_dir}/dev_document_tok.jsonl') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = json.loads(line)
                index, toks, idxs = line
                bert_batch.append(toks)
                bert_indexs.append(idxs)
        phrase_reps_list, all_phrases = self.encode_doc_ids_batch(bert_batch, bert_indexs, batch_size=80, quiet=quiet)
        phrase_reps = torch.vstack(phrase_reps_list)
        mean = torch.mean(phrase_reps, dim=0)
        std = torch.std(phrase_reps, dim=0)
        phrase_num = phrase_reps.shape[0]
        return phrase_num, mean, std

    @torch.no_grad()
    def encode_doc_batch(self, sentences, batch_size=128, device='cuda'):
        all_s_rep = []
        all_e_rep = []
        all_offsets = []
        for st_idx in range(0, len(sentences), batch_size):
            end_idx = min(len(sentences), st_idx + batch_size)
            batch_sentences = sentences[st_idx: end_idx]
            encoded_dict = self.bert_tokenizer(
                batch_sentences,
                max_length=512,
                return_tensors='pt',
                padding='longest',
                return_offsets_mapping=True
            )
            offset_mapping = encoded_dict['offset_mapping']
            all_offsets.extend(offset_mapping)

            encoded_dict = {k: v.to(device) for k, v in encoded_dict.items()}
            output = self.phrase_encoder(**{k: v for k, v in encoded_dict.items() if k != 'offset_mapping'}, output_hidden_states=True)['hidden_states'][-1]  # [B, S, E]
            # collect the phrase start representations and phrase end representations
            s_rep = self.s_proj(output)
            e_rep = self.e_proj(output)
            all_s_rep.extend(s_rep)
            all_e_rep.extend(e_rep)
        return all_s_rep, all_e_rep, all_offsets

    @torch.no_grad()
    def encode_query_batch(self, sentences, batch_size=256, device='cuda'):
        self.tokenizer.pad_token = self.tokenizer.eos_token_id
        all_rep = []
        all_offsets = []
        for st_idx in range(0, len(sentences), batch_size):
            end_idx = min(len(sentences), st_idx + batch_size)
            batch_sentences = sentences[st_idx: end_idx]
            encoded_dict = self.tokenizer(
                batch_sentences,
                max_length=512,
                return_tensors='pt',
                padding='longest',
                return_offsets_mapping=True
            )

            encoded_dict = {k: v.to(device) for k, v in encoded_dict.items()}
            offset_mapping = encoded_dict['offset_mapping']
            all_offsets.extend(offset_mapping)

            output = self.model(**{k: v for k, v in encoded_dict.items() if k != 'offset_mapping'}, output_hidden_states=True)['hidden_states'][-1]  # [B, S, E]
            # collect the phrase start representations and phrase end representations
            all_rep.extend(output)
        return all_rep, all_offsets