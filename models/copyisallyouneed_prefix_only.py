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
                self.args['prefix_encoder_model'][self.args['lang']][self.args['model_size']]
            )
            self.model = GPT2LMHeadModel(GPT_config)
            # self.model = AutoModel.from_config(GPT_config)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(self.args['prefix_encoder_model'][self.args['lang']][self.args['model_size']])
        # GPT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args['prefix_encoder_tokenizer'][self.args['lang']][self.args['model_size']])

        self.vocab_size = len(self.tokenizer)
        self.pad = self.tokenizer.pad_token_id if self.args['lang'] == 'zh' else self.tokenizer.eos_token_id
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        assert self.token_embeddings.shape[0] == self.vocab_size
        
        self.phrase_dim = self.args['phrase_dim'] if 'phrase_dim' in self.args else self.model.config.hidden_size
        if args['sep_proj']:
            self.token_proj = nn.Linear(self.model.config.hidden_size, self.phrase_dim)
            self.prefix_proj = nn.Linear(self.model.config.hidden_size, self.phrase_dim)
        else:
            self.dim_proj = nn.Linear(self.model.config.hidden_size, self.phrase_dim)
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.phrase_norm_mean = 0.3515
        self.phrase_embedding = np.memmap('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/phrase_embedding_index/PCA_emb_merged_memmap.npy', dtype=np.float32, mode='r', shape=(137101097, 128))

    @torch.no_grad()
    def get_query_rep(self, ids):
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        if self.args['sep_proj']:
            output = self.prefix_proj(output)
        else:
            output = self.dim_proj(output)
        return output
    
    @torch.no_grad()
    def get_query_rep_batch(self, ids):
        all_length = [len(x) for x in ids]
        all_gpt_ids = pad_sequence([torch.LongTensor(x) for x in ids], padding_value=self.tokenizer.eos_token_id, batch_first=True).cuda()
        all_gpt2_mask = generate_mask(all_gpt_ids, pad_token_idx=self.tokenizer.eos_token_id).cuda()

        output = self.model(input_ids=all_gpt_ids, attention_mask=all_gpt2_mask, output_hidden_states=True)['hidden_states'][-1]
        output_mask = torch.LongTensor(all_length) - 1
        output = output[range(len(ids)), output_mask]
        if self.args['sep_proj']:
            output = self.prefix_proj(output)
        else:
            output = self.dim_proj(output)
        return output

    def get_token_reps(self):
        if self.args['sep_proj']:
            output = self.token_proj(self.token_embeddings)
        else:
            output = self.dim_proj(self.token_embeddings)
        return output

    def get_token_loss(self, ids, hs, mask, token_reps, do_eval=True):
        mask = mask.to(torch.bool)
        label = ids[:, 1:].reshape(-1)
        label_mask = mask[:, 1:].reshape(-1)
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
        # model_st_time = time()
        token_reps = self.token_embeddings
        if self.args['sep_proj']:
            token_reps = self.token_proj(token_reps)
        else:
            token_reps = self.dim_proj(token_reps)
        
        ## gpt2 query encoder
        ids, ids_mask = batch['gpt2_ids'].cuda(), batch['gpt2_mask'].cuda()
        query_hidden_states = \
        self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True).hidden_states[-1]
        if self.args['sep_proj']:
            query_hidden_states = self.prefix_proj(query_hidden_states)
        else:
            query_hidden_states = self.dim_proj(query_hidden_states)
        AR_loss = self.get_token_loss(ids, query_hidden_states, ids_mask, token_reps, do_eval=False)
        # print('AR_loss_acc:', AR_loss, acc)

        phrase_emb_idx_list = batch['phrase_embedding_idx_list']
        if not phrase_emb_idx_list:
            return AR_loss, {}

        phrase_rep = torch.from_numpy(self.phrase_embedding[phrase_emb_idx_list]).cuda()
        candidate_reps = torch.cat([
            token_reps,
            phrase_rep], dim=0)
    
        labels = batch['phrase_label'].cuda()
        query = query_hidden_states[:, :-1].reshape(-1, query_hidden_states.shape[-1])
        query_mask = ids_mask[:, 1:].reshape(-1).to(torch.bool)
        query = query[query_mask]
        labels = labels[:, 1:].reshape(-1)
        labels = labels[query_mask]
        phrase_indexes = labels >= self.vocab_size
        token_indexes = ~phrase_indexes
        # print('query:', query.shape, labels.shape, phrase_indexes.sum().item())
        phrase_num = phrase_indexes.sum()
        token_num = token_indexes.sum()
        phrase_batch_size = 20000
        logits = []
        for phrase_st in range(0, len(candidate_reps), phrase_batch_size):
            phrase_end = min(phrase_st + phrase_batch_size, len(candidate_reps))
            logits_ = torch.matmul(query, candidate_reps[phrase_st: phrase_end].t())
            logits_ /= self.args['temp']
            logits.append(logits_)
        logits = torch.hstack(logits)

        false_neg_mask = batch['false_neg_mask']
        if false_neg_mask is not None:
            false_neg_mask = false_neg_mask.cuda()
            false_neg_mask = false_neg_mask[:, 1:].reshape(-1, false_neg_mask.shape[-1])
            false_neg_mask = false_neg_mask[query_mask]
            # print(false_neg_mask.shape, logits.shape)
            logits = torch.where(false_neg_mask.to(torch.bool), logits, torch.tensor(-np.inf).to(torch.half).cuda())
        if torch.isnan(logits).any():
            # phrase_list = batch['phrase_list']
            sys.stderr.write(f"logits nan\n")
            sys.stderr.write(f"logits: {logits}\n")
            nan_pos = torch.nonzero(torch.isnan(logits)).squeeze()
            sys.stderr.write(f"nan pos: {nan_pos}\n")
            # if not nan_pos.shape[0] == 0:
            #     if nan_pos[0, 0].item() >= self.vocab_size:
            #         sys.stderr.write(f'error phrase: {phrase_list[nan_pos[0, 0].item() - self.vocab_size]}')
            sys.stderr.write(f"query has nan: {torch.isnan(query).any()}\n")
            sys.stderr.write(f"query nan pos: {torch.nonzero(torch.isnan(query)).squeeze()}\n")
            sys.stderr.write(f"token emb has nan: {torch.isnan(token_reps).any()}\n")
            sys.stderr.write(f"token emb nan pos: {torch.nonzero(torch.isnan(token_reps)).squeeze()}\n")
            # sys.stderr.write(f"query: {self.tokenizer.decode(ids.reshape(-1))}\n")
            sys.stderr.write(f"phrase num: {len(phrase_rep)}\n")
            return AR_loss, {}
        if self.args['loss_type'] == 'focal_loss':
            logit_logsoftmax = F.log_softmax(logits, dim=-1)
            logit_softmax = torch.exp(logit_logsoftmax)
            p = logit_softmax[range(len(logits)), labels]
            logp = logit_logsoftmax[range(len(logits)), labels]
            alpha = 0.25
            gamma = self.args['gamma']
            loss_ = -alpha * (1 - p) ** gamma * logp
        else:
            loss_ = F.log_softmax(logits, dim=-1)
            loss_ = -loss_[range(len(logits)), labels].sum(dim=-1)
        # loss = loss_.mean()
        beta = self.args['beta']
        loss = 0
        if phrase_num > 0:
            loss += beta * loss_[phrase_indexes].mean()
        if token_num > 0:
            loss += (1 - beta) * loss_[token_indexes].mean()
        if torch.isnan(loss).any():
            # phrase_list = batch['phrase_list']
            sys.stderr.write(f"loss nan\n")
            sys.stderr.write(f"loss: {loss}\n")
            nan_pos = torch.nonzero(torch.isnan(logits)).squeeze()
            sys.stderr.write(f"nan pos: {nan_pos}\n")
            # if not nan_pos.shape[0] == 0:
            #     if nan_pos[0, 0].item() >= self.vocab_size:
            #         sys.stderr.write(f'error phrase: {phrase_list[nan_pos[0, 0].item() - self.vocab_size]}')
            sys.stderr.write(f"query has nan: {torch.isnan(query).any()}\n")
            sys.stderr.write(f"query nan pos: {torch.nonzero(torch.isnan(query)).squeeze()}\n")
            sys.stderr.write(f"token emb has nan: {torch.isnan(token_reps).any()}\n")
            sys.stderr.write(f"token emb nan pos: {torch.nonzero(torch.isnan(token_reps)).squeeze()}\n")
            sys.stderr.write(f"query: {self.tokenizer.decode(ids.reshape(-1))}\n")
            sys.stderr.write(f"phrase num: {len(phrase_rep)}\n")
            return AR_loss, {}
        loss += AR_loss
        if self.args['l2_penalty']:
            token_norm = torch.norm(token_reps, dim=-1)
            l2_penalty = token_norm.mean() - self.phrase_norm_mean
            loss += l2_penalty
        result_dict = {}
        # result_dict = {
        #     'phrase_acc': phrase_acc,
        #     'token_acc': token_acc
        # }
        # model_end_time = time()
        # print(f"model worker{self.args['local_rank']} time:", model_end_time - model_st_time, model_end_time - t2, t2- t1, t1 - model_st_time)
        return (
            loss,  # phrase(tok) classification loss
            result_dict
        )

    @torch.no_grad()
    def evaluate(self, quiet=True):
        self.model.eval()

        data_fn = f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/minipile/match_result_tok/validation_tok_small'
        with open(data_fn) as f:
            dev_data_queue = f.readlines()
        # 966 (1024)
        dev_data_queue = [json.loads(x) for x in dev_data_queue]
        dev_data_queue.sort(key=lambda x: len(x[0]), reverse=True)

        all_gpt_ids, all_valid_phrases, all_document, all_global_suffix_st_char, all_length = [], [], [], [], []
        for gpt_ids, valid_phrases, document, global_suffix_st_char, *_ in dev_data_queue:
            all_gpt_ids.append(gpt_ids)
            all_length.append(len(gpt_ids))
            all_valid_phrases.append(valid_phrases)
            all_document.append(document)
            all_global_suffix_st_char.append(global_suffix_st_char)

        phrase_list = []
        emb_idx_map = {}
        emb_idx_list = []
        for instance_idx, valid_phrases in enumerate(all_valid_phrases):
            for phrase, start_idx, end_idx, ref_pos in valid_phrases:
                if ref_pos not in emb_idx_map:
                    emb_idx_map[ref_pos] = len(emb_idx_list)
                    emb_idx_list.append(ref_pos)
                    phrase_list.append(phrase)
        print(f'phrase_list:', len(phrase_list))
        if emb_idx_list:
            embeddings = torch.from_numpy(self.phrase_embedding[emb_idx_list])
        else:
            embeddings = None
        
        batch_size = 4
        topk = 100
        k_list = [0, 1, 3, 5, 10, 20, 100]
        max_suffix_length = 72
        embeddings = embeddings.cuda()
        tok_reps = self.token_embeddings
        if self.args['sep_proj']:
            tok_reps = self.token_proj(tok_reps)
        else:
            tok_reps = self.dim_proj(tok_reps)
        candidate_reps = torch.vstack([tok_reps, embeddings])
        # print(candidate_reps.shape) # torch.Size([70063, 128])

        all_result = collections.defaultdict(int)
        tok_counter, phrase_counter = 0, 0
        for batch_st in tqdm(range(0, len(all_gpt_ids), batch_size), disable=quiet):
            batch_end = min(batch_st + batch_size, len(all_gpt_ids))
            ids_cpu = all_gpt_ids[batch_st: batch_end]
            batch_length = all_length[batch_st: batch_end]
            batch_valid_phrases = all_valid_phrases[batch_st: batch_end]
            batch_documents = all_document[batch_st: batch_end]
            batch_global_suffix_st_char = all_global_suffix_st_char[batch_st: batch_end]

            ids_cpu = pad_sequence([torch.LongTensor(x) for x in ids_cpu], padding_value=self.tokenizer.eos_token_id, batch_first=True)
            gpt2_mask = generate_mask(ids_cpu, pad_token_idx=self.tokenizer.eos_token_id)
            phrase_label = deepcopy(ids_cpu)
            for local_idx, valid_phrases in enumerate(batch_valid_phrases):
                for phrase, start_idx, end_idx, ref_pos in valid_phrases:
                    emb_idx = emb_idx_map[ref_pos]
                    phrase_label[local_idx][start_idx] = len(self.tokenizer) + emb_idx
            ids = ids_cpu.cuda()
            gpt2_mask = gpt2_mask.cuda()
            gpt_reps = \
            self.model(input_ids=ids, attention_mask=gpt2_mask, output_hidden_states=True).hidden_states[-1]
            if self.args['sep_proj']:
                gpt_reps = self.prefix_proj(gpt_reps)
            else:
                gpt_reps = self.dim_proj(gpt_reps)
            AR_loss, acc = self.get_token_loss(ids, gpt_reps, gpt2_mask, tok_reps, do_eval=True)
            logits_ = torch.matmul(gpt_reps, candidate_reps.t())
            topk_logits, topk_indexs = torch.topk(logits_, k=topk, dim=-1)
            topk_indexs = topk_indexs.cpu()
            for instance_idx in range(len(topk_indexs)):
                tokens, topk_pred, label = ids_cpu[instance_idx][1:].tolist(), topk_indexs[instance_idx][:-1].tolist(), phrase_label[instance_idx][1:].tolist()
                length = batch_length[instance_idx]
                document = batch_documents[instance_idx]
                suffix_st_char_list = batch_global_suffix_st_char[instance_idx][1:]
                for tok_idx, (tok_, topk_pred_, label_, suffix_st_char_) in enumerate(zip(tokens, topk_pred, label, suffix_st_char_list)):
                    if tok_idx >= length - 1:
                        break
                    if label_ < self.vocab_size: # token
                        # print(self.tokenizer.decode(tok_), topk_pred_, label_, self.tokenizer.decode(label_))
                        tok_counter += 1
                        for k_idx in range(len(k_list) - 1):
                            st, end = k_list[k_idx], k_list[k_idx + 1]
                            if label_ in topk_pred_[st: end]:
                                for later_end in range(k_idx + 1, len(k_list)):
                                    all_result[f'token_hit@{k_list[later_end]}'] += 1
                                    all_result[f'global_acc@{k_list[later_end]}'] += 1
                                    all_result[f'global_hit@{k_list[later_end]}'] += 1
                                break
                    else:
                        # print(self.tokenizer.decode(tok_), topk_pred_, label_, phrase_list[label_ - self.vocab_size])
                        phrase_counter += 1
                        # hit
                        for k_idx in range(len(k_list) - 1):
                            st, end = k_list[k_idx], k_list[k_idx + 1]
                            if label_ in topk_pred_[st: end]:
                                for later_end in range(k_idx + 1, len(k_list)):
                                    all_result[f'phrase_hit@{k_list[later_end]}'] += 1
                                    all_result[f'global_hit@{k_list[later_end]}'] += 1
                                break
                        suffix_ = document[suffix_st_char_: suffix_st_char_ + max_suffix_length].lstrip() + ' '
                        # print(suffix_)

                        # target_phrase = phrase_list[label_ - self.vocab_size]
                        # # print('target_phrase:', target_phrase)

                        # recall & acc
                        # valid_counter = 0
                        for idx in range(topk):
                            pred_idx_ = topk_pred_[idx]
                            if (pred_idx_ >= self.vocab_size and suffix_.startswith(phrase_list[pred_idx_ - self.vocab_size] + ' ')) \
                                or (pred_idx_ < self.vocab_size and pred_idx_ == tok_):
                                # if random.random() < 0.001:
                                #     pred_phrase_ = phrase_list[pred_idx_ - self.vocab_size] if pred_idx_ >= self.vocab_size else self.tokenizer.decode(pred_idx_)
                                #     print(self.tokenizer.decode(tokens[max(0, tok_idx-9): tok_idx+1]), '|||', pred_idx_, '|||', pred_phrase_, '|||', label_, '|||', phrase_list[label_ - self.vocab_size])
                                
                                # valid_counter += 1
                                # if valid_counter == 1:
                                for tmp in range(idx, topk):
                                    if tmp + 1 in k_list[1:]: 
                                        all_result[f'phrase_acc@{tmp + 1}'] += 1
                                        all_result[f'global_acc@{tmp + 1}'] += 1
                                break
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