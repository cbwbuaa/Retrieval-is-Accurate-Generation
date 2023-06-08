from header import *
import sys
import numpy as np
from time import time
# torch.set_printoptions(profile='full')

class CopyisallyouneedAllRefV2(nn.Module):

    def __init__(self, **args):
        super(CopyisallyouneedAllRefV2, self).__init__()
        self.args = args

        # bert-encoder model
        self.phrase_encoder = AutoModel.from_pretrained(
            self.args['phrase_encoder_model'][self.args['lang']]
        )
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            self.args['phrase_encoder_tokenizer'][self.args['lang']]
        )
        self.bert_tokenizer.add_tokens(['<|endoftext|>'])
        self.phrase_encoder.resize_token_embeddings(len(self.bert_tokenizer))

        # model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(self.args['prefix_encoder_model'][self.args['lang']])
        self.tokenizer = AutoTokenizer.from_pretrained(self.args['prefix_encoder_tokenizer'][self.args['lang']])
        self.tokenizer.add_tokens(['<|endoftext|>'])
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.vocab_size = len(self.tokenizer)
        self.pad = self.tokenizer.pad_token_id if self.args['lang'] == 'zh' else self.tokenizer.bos_token_id
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        assert self.token_embeddings.shape[0] == self.vocab_size
        # MLP: mapping bert phrase start representations
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size // 2)
        )
        # MLP: mapping bert phrase end representations
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size // 2)
        )
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)

        self.pre_batch_phrases = []
        self.pre_batch_phrases_time_cnt = []
        self.pre_batch_step_limit = args["prebatch_phrase_num"] if 'prebatch_phrase_num' in args else 5

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return output

    def _update_prebatch_phrase(self, cur_phrase_emb):
        self.pre_batch_phrases.append(cur_phrase_emb.detach())
        self.pre_batch_phrases = self.pre_batch_phrases[-self.pre_batch_step_limit:]

    def _get_prebatch_phrase_emb(self):
        return torch.vstack(self.pre_batch_phrases) if self.pre_batch_phrases else None

    def forward(self, batch):    
        # model_st_time = time()    
        # # sanity check
        # ids, ids_mask, labels, label_mask = batch['gpt2_ids'], batch['gpt2_mask'], batch['phrase_labels'], batch['gpt2_label_mask']
        # ids = ids[:, :-1].reshape(-1)
        # ids_mask = ids_mask[:, :-1].reshape(-1).to(torch.bool)
        # label_mask = label_mask[:, :-1].reshape(-1).to(torch.bool)
        # query = ids[ids_mask]
        # hs = ids[label_mask]
        # print(len(query), '[!]', self.tokenizer.decode(query))
        # print(len(hs), '[!]', self.tokenizer.decode(hs))
        # dids, dids_mask = batch['bert_ids'], batch['bert_mask']
        # dids = dids.reshape(-1)
        # start_labels, end_labels = batch['start_labels'].reshape(-1).cpu().tolist(), batch['end_labels'].reshape(-1).cpu().tolist() # make sure do have prefix
        # in_doc_neg_start_labels = batch['in_doc_neg_start_labels'].reshape(-1).cpu().tolist()
        # in_doc_neg_end_labels = batch['in_doc_neg_end_labels'].reshape(-1).cpu().tolist()
        # all_phrases = []
        # for st, end in zip(start_labels, end_labels):
        #     if st >= self.vocab_size:
        #         target = dids[st - self.vocab_size: end + 1 - self.vocab_size]
        #         all_phrases.append(self.bert_tokenizer.decode(target)) 
        #         # print('target:', self.bert_tokenizer.decode(dids[:st - self.vocab_size]), '***', self.bert_tokenizer.decode(target), '***', self.bert_tokenizer.decode(dids[end + 1 - self.vocab_size:]))
        # print('target:', all_phrases)
        # print(len(labels.reshape(-1)), labels.reshape(-1) - self.vocab_size)
        # all_phrases = []
        # for st, end in zip(in_doc_neg_start_labels, in_doc_neg_end_labels):
        #     target = dids[st: end + 1]
        #     all_phrases.append(self.bert_tokenizer.decode(target))
        #     # print('neg:', self.bert_tokenizer.decode(dids[:st]), '***', self.bert_tokenizer.decode(target), '***', self.bert_tokenizer.decode(dids[end + 1:]))
        # print('canidates:', all_phrases)
        # exit()
        
        ## encode the document with the BERT encoder model
        dids, dids_mask = batch['bert_ids'], batch['bert_mask']
        candidate_reps = self.token_embeddings
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]  # [B, S, E]
        # collect the phrase start representations and phrase end representations
        s_rep = self.s_proj(output)
        e_rep = self.e_proj(output)
        s_rep = s_rep.reshape(-1, s_rep.size(-1))
        e_rep = e_rep.reshape(-1, e_rep.size(-1))  # [B_doc*S_doc, 768//2]
    
        start_labels, end_labels = batch['start_labels'].reshape(-1), batch['end_labels'].reshape(-1)

        phrase_label_mask = start_labels >= self.vocab_size
        doc_phrase_st_rep = s_rep[start_labels[phrase_label_mask] - self.vocab_size]
        doc_phrase_end_rep = e_rep[end_labels[phrase_label_mask]- self.vocab_size]
        doc_phrase_rep = torch.cat([doc_phrase_st_rep, doc_phrase_end_rep], dim=-1)
        
        in_doc_neg_start_labels = batch['in_doc_neg_start_labels'].reshape(-1)
        in_doc_neg_end_labels = batch['in_doc_neg_end_labels'].reshape(-1)
        in_doc_st_rep = s_rep[in_doc_neg_start_labels]
        in_doc_end_rep = e_rep[in_doc_neg_end_labels]
        in_doc_rep = torch.cat([in_doc_st_rep, in_doc_end_rep], dim=-1)
        
        phrase_rep = torch.cat([
            doc_phrase_rep,
            in_doc_rep], dim=0)
        candidate_reps = torch.cat([
            candidate_reps,
            phrase_rep], dim=0)

        prebatch_emb = self._get_prebatch_phrase_emb()
        if prebatch_emb is not None:
            candidate_reps = torch.cat([
                candidate_reps,
                prebatch_emb], dim=0)

        ## gpt2 query encoder
        ids, ids_mask, label_mask, labels = batch['gpt2_ids'], batch['gpt2_mask'], batch['gpt2_label_mask'], batch['phrase_labels']
        last_hidden_states = \
        self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True).hidden_states[-1]

        # collect the query representations
        query = last_hidden_states[:, :-1].reshape(-1, last_hidden_states.shape[-1])
        label_mask = label_mask[:, 1:].reshape(-1).to(torch.bool)
        query = query[label_mask]
        labels = labels.reshape(-1)
        phrase_indexes = labels >= self.vocab_size
        token_indexes = ~phrase_indexes
        phrase_num = phrase_indexes.sum()
        token_num = token_indexes.sum()
        if self.args['normalize']:
            logits = torch.matmul(query / query_norm, (candidate_reps / torch.norm(candidate_reps, p=2, dim=-1, keepdim=True)).t())
        else:
            logits = torch.matmul(query, candidate_reps.t())
        logits /= self.args['temp']
        false_neg_mask = batch['false_neg_mask']
        false_neg_mask = false_neg_mask.reshape(-1, false_neg_mask.shape[-1])
        # print(false_neg_mask.shape, logits.shape)
        logits = torch.where(false_neg_mask.to(torch.bool), logits, torch.tensor(-np.inf).to(torch.half).cuda())

        if torch.isnan(logits).any():
            sys.stderr.write(f"logits nan\n")
            sys.stderr.write(f"logits: {logits}\n")
            sys.stderr.write(f"query: {self.tokenizer.decode(ids.reshape(-1))}\n")
            sys.stderr.write(f"doc: {self.bert_tokenizer.decode(dids.reshape(-1))}\n")
            sys.stderr.write(f"labels: {labels}\n")
            exit()
        ## split the token accuaracy and phrase accuracy
        phrase_acc = logits[phrase_indexes].max(dim=-1)[1] == labels[phrase_indexes]
        phrase_acc = phrase_acc.to(torch.float).mean().item()
        if token_num > 0: # no token
            token_acc = logits[token_indexes].max(dim=-1)[1] == labels[token_indexes]
            token_acc = token_acc.to(torch.float).mean().item()
        else:
            token_acc = -1
        if self.args['loss_type'] == 'focal_loss':
            logit_logsoftmax = F.log_softmax(logits, dim=-1)
            logit_softmax = torch.exp(logit_logsoftmax)
            p = logit_softmax[range(len(logits)), labels]
            logp = logit_logsoftmax[range(len(logits)), labels]
            alpha = 0.25
            gamma = 2
            loss_ = -alpha * (1 - p) ** gamma * logp
            beta = self.args['beta']
            loss = 0
            if phrase_num > 0:
                loss += beta * loss_[phrase_indexes].mean()
            if token_num > 0:
                loss += (1 - beta) * loss_[token_indexes].mean()
        else:
            loss = F.log_softmax(logits, dim=-1)
            loss = loss[range(len(logits)), labels]
            loss = (-loss.sum(dim=-1)).mean()
        if torch.isnan(loss).any():
            sys.stderr.write(f"loss nan\n")
            sys.stderr.write(f"loss: {loss}\n")
            sys.stderr.write(f"logits: {logits}\n")
            sys.stderr.write(f"query: {self.tokenizer.decode(ids.reshape(-1))}\n")
            sys.stderr.write(f"doc: {self.bert_tokenizer.decode(dids.reshape(-1))}\n")
            sys.stderr.write(f"labels: {labels}\n")
            exit()
        result_dict = {
            'phrase_acc': phrase_acc,
            'token_acc': token_acc
        }
        self._update_prebatch_phrase(phrase_rep)
        # print("model time:", time() - model_st_time)
        return (
            loss,  # phrase(tok) classification loss
            result_dict
        )

    @torch.no_grad()
    def encode_doc_batch(self, sentences, batch_size=256, device='cuda'):
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
            encoded_dict = {k: v.to(device) for k, v in encoded_dict.items()}
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