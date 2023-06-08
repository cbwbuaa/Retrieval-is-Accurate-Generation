from header import *
import sys
# torch.set_printoptions(profile='full')

class CopyisallyouneedAllRef(nn.Module):

    def __init__(self, **args):
        super(CopyisallyouneedAllRef, self).__init__()
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

    def update_prebatch_phrase(self):
        self.pre_batch_phrases_time_cnt = [x + 1 for x in self.pre_batch_phrases_time_cnt]
        while self.pre_batch_phrases_time_cnt[0] >= self.pre_batch_step_limit:
            self.pre_batch_phrases.pop(0)
            self.pre_batch_phrases_time_cnt.pop(0)

    def _add_prebatch_phrase(self, cur_phrase_emb, cur_phrase_text):
        for idx in range(len(cur_phrase_text)):
            emb = cur_phrase_emb[idx]
            text = cur_phrase_text[idx]
            self.pre_batch_phrases.append((emb.detach(), text))
            self.pre_batch_phrases_time_cnt.append(0)

    def _get_prebatch_phrase_emb(self, cur_phrase_text):
        embs = [emb for emb, text in self.pre_batch_phrases if text not in cur_phrase_text]
        return torch.vstack(embs) if embs else None

    def _get_token_loss(self, ids, hs, ids_mask):
        # no pad token
        label = ids[:, 1:]
        logits = torch.matmul(
            hs[:, :-1, :],
            self.token_embeddings.t()
        )
        # TODO: inner loss function remove the temperature factor
        logits /= self.args['temp']
        loss = self.gen_loss_fct(logits.view(-1, logits.size(-1)), label.reshape(-1))
        chosen_tokens = torch.max(logits, dim=-1)[1]
        gen_acc = (chosen_tokens.reshape(-1) == label.reshape(-1)).to(torch.long)
        valid_mask = (label != self.pad).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc

    def forward(self, batch):
        ## encode the document with the BERT encoder model
        dids, dids_mask = batch['bert_ids'], batch['bert_mask']
        
        # # sanity check
        # ids, ids_mask, labels = batch['gpt2_ids'], batch['gpt2_mask'], batch['phrase_labels'][0]
        # print(self.tokenizer.decode(ids.view(-1)))
        # dids = dids.view(-1)
        # start_labels, end_labels = batch['start_labels'][0].reshape(-1).cpu().tolist(), batch['end_labels'][0].reshape(-1).cpu().tolist() # make sure do have prefix
        # in_doc_neg_start_labels = batch['in_doc_neg_start_labels'][0].cpu().tolist()
        # in_doc_neg_end_labels = batch['in_doc_neg_end_labels'][0].cpu().tolist()
        # all_phrases = []
        # for st, end in zip(start_labels, end_labels):
        #     if st >= self.vocab_size:
        #         target = dids[st - self.vocab_size: end + 1 - self.vocab_size]
        #         # all_phrases.append(self.bert_tokenizer.decode(target)) 
        #         print('target:', self.bert_tokenizer.decode(dids[:st - self.vocab_size]), '***', self.bert_tokenizer.decode(target), '***', self.bert_tokenizer.decode(dids[end + 1 - self.vocab_size:]))
        # # print('target:', all_phrases)
        # all_phrases = []
        # for st, end in zip(in_doc_neg_start_labels, in_doc_neg_end_labels):
        #     target = dids[st: end + 1]
        #     all_phrases.append(self.bert_tokenizer.decode(target))
        #     print('neg:', self.bert_tokenizer.decode(dids[:st]), '***', self.bert_tokenizer.decode(target), '***', self.bert_tokenizer.decode(dids[end + 1:]))
        # print('canidates:', all_phrases)
        # exit()

        candidate_reps = self.token_embeddings
        # norm
        token_norm = torch.norm(self.token_embeddings, p=2, dim=-1, keepdim=True).mean().item()
        if dids.size(-1) > 0:
            output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]  # [B, S, E]
            # collect the phrase start representations and phrase end representations
            s_rep = self.s_proj(output)
            e_rep = self.e_proj(output)
            s_rep = s_rep.reshape(-1, s_rep.size(-1))
            e_rep = e_rep.reshape(-1, e_rep.size(-1))  # [B_doc*S_doc, 768//2]
            # norm
            doc_norm = torch.norm(torch.hstack((s_rep, e_rep)), p=2, dim=-1, keepdim=True).mean().item()

            start_labels, end_labels = batch['start_labels'].reshape(-1), batch['end_labels'].reshape(-1)
            phrase_label_mask = start_labels >= self.vocab_size
            doc_phrase_st_rep = s_rep[start_labels[phrase_label_mask] - self.vocab_size]
            doc_phrase_end_rep = e_rep[end_labels[phrase_label_mask]- self.vocab_size]
            doc_phrase_rep = torch.cat([doc_phrase_st_rep, doc_phrase_end_rep], dim=-1)
            # norm
            phrase_norm = torch.norm(doc_phrase_rep, p=2, dim=-1, keepdim=True).mean().item()
            candidate_reps = torch.cat([
                candidate_reps,
                doc_phrase_rep], dim=0)
            
            if self.args['in_doc_neg_num'] != 0:
                in_doc_neg_start_labels = batch['in_doc_neg_start_labels'][0]
                in_doc_neg_end_labels = batch['in_doc_neg_end_labels'][0]
                in_doc_st_rep = s_rep[in_doc_neg_start_labels]
                in_doc_end_rep = e_rep[in_doc_neg_end_labels]
                in_doc_rep = torch.cat([in_doc_st_rep, in_doc_end_rep], dim=-1)
                candidate_reps = torch.cat([
                    candidate_reps,
                    in_doc_rep], dim=0)
        else:
            doc_norm = -1
            phrase_norm = -1

        prebatch_emb = self._get_prebatch_phrase_emb(batch['phrases'])
        if prebatch_emb is not None:
            candidate_reps = torch.cat([
                candidate_reps,
                prebatch_emb], dim=0)

        ## gpt2 query encoder
        ids, ids_mask, labels = batch['gpt2_ids'], batch['gpt2_mask'], batch['phrase_labels'][0]
        assert ids.shape[0] == 1
        last_hidden_states = \
        self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True).hidden_states[-1]
        last_hidden_states = last_hidden_states[0]

        # collect the query representations
        query = last_hidden_states[-labels.shape[-1] - 1: -1]
        # norm
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True)
        query_norm_mean = query_norm.mean().item()

        phrase_indexes = labels >= self.vocab_size
        token_indexes = ~phrase_indexes
        phrase_num = phrase_indexes.sum()
        token_num = token_indexes.sum()
        if self.args['normalize']:
            logits = torch.matmul(query / query_norm, (candidate_reps / torch.norm(candidate_reps, p=2, dim=-1, keepdim=True)).t())
            cosine = logits
            token_IP, phrase_IP = -1, -1
            token_cos = cosine[:, :self.vocab_size].mean().item()
            phrase_cos = cosine[:, self.vocab_size:].mean().item()
        else:
            logits = torch.matmul(query, candidate_reps.t())
            cosine = torch.matmul(query / query_norm, (candidate_reps / torch.norm(candidate_reps, p=2, dim=-1, keepdim=True)).t())
            token_IP = logits[:, :self.vocab_size].mean().item()
            phrase_IP = logits[:, self.vocab_size:].mean().item()
            token_cos = cosine[:, :self.vocab_size].mean().item()
            phrase_cos = cosine[:, self.vocab_size:].mean().item()
        logits /= self.args['temp']
        if torch.isnan(logits).any():
            sys.stderr.write(f"logits nan\n")
            sys.stderr.write(f"logits: {logits}\n")
            sys.stderr.write(f"query: {self.tokenizer.decode(ids.view(-1))}\n")
            sys.stderr.write(f"doc: {self.bert_tokenizer.decode(dids.view(-1))}\n")
            sys.stderr.write(f"labels: {labels}\n")
            exit()
        ## split the token accuaracy and phrase accuracy
        if phrase_num > 0: # no phrase
            phrase_acc = logits[phrase_indexes].max(dim=-1)[1] == labels[phrase_indexes]
            wrong_phrase_to_token = logits[phrase_indexes].max(dim=-1)[1] < self.vocab_size
            wrong_phrase_to_token = wrong_phrase_to_token.to(torch.float).mean().item()
            phrase_acc = phrase_acc.to(torch.float).mean().item()
            phrase_inner_acc = (logits[phrase_indexes, self.vocab_size:].max(dim=-1)[1] + self.vocab_size) == labels[phrase_indexes]
            phrase_inner_acc = phrase_inner_acc.to(torch.float).mean().item()
            valid_phrase_texts = batch['phrases']
            self._add_prebatch_phrase(doc_phrase_rep, valid_phrase_texts)
        else:
            phrase_acc = -1
            wrong_phrase_to_token = -1
            phrase_inner_acc = -1
        if token_num > 0: # no token
            token_acc = logits[token_indexes].max(dim=-1)[1] == labels[token_indexes]
            token_acc = token_acc.to(torch.float).mean().item()
            token_inner_acc = logits[token_indexes, :self.vocab_size].max(dim=-1)[1] == labels[token_indexes]
            token_inner_acc = token_inner_acc.to(torch.float).mean().item()
        else:
            token_acc = -1
            token_inner_acc = -1

        # prevent token & phrase competition
        query_ids = ids[0, -labels.shape[-1]: ]
        phrase_mask = labels >= self.vocab_size
        phrase_idx = torch.nonzero(phrase_mask.to(torch.long)).view(-1)
        # # sanity check
        # print(ids[0])
        # print(query_ids)
        # print(phrase_mask)
        # print(phrase_idx)
        # print(query_ids[phrase_mask])
        logits[phrase_idx, query_ids[phrase_mask]] = -1e4
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), labels] = 1.
        if self.args['loss_type'] == 'focal_loss':
            logit_logsoftmax = F.log_softmax(logits, dim=-1)
            logit_softmax = torch.exp(logit_logsoftmax)
            p = (logit_softmax * mask).sum(dim=-1)
            logp = (logit_logsoftmax * mask).sum(dim=-1)
            alpha = 0.25
            gamma = 2
            loss_ = -alpha * (1 - p) ** gamma * logp
            beta = self.args['beta']
            loss = 0
            if phrase_num > 0:
                loss += beta * loss_[phrase_indexes].mean()
            if token_num > 0:
                loss += (1 - beta) * loss_[token_indexes].mean()
        elif self.args['loss_type'] == 'balance_loss':
            loss = F.log_softmax(logits, dim=-1) * mask
            # balance token and phrase loss
            if phrase_num > 0:
                loss[phrase_indexes] /= phrase_num
            if token_num > 0:
                loss[token_indexes] /= token_num
            loss = (-loss.sum(dim=-1)).mean()
        else:
            loss = F.log_softmax(logits, dim=-1) * mask
            loss = (-loss.sum(dim=-1)).mean()
        if torch.isnan(loss).any():
            sys.stderr.write(f"loss nan\n")
            sys.stderr.write(f"loss: {loss}\n")
            sys.stderr.write(f"logits: {logits}\n")
            sys.stderr.write(f"query: {self.tokenizer.decode(ids.view(-1))}\n")
            sys.stderr.write(f"doc: {self.bert_tokenizer.decode(dids.view(-1))}\n")
            sys.stderr.write(f"labels: {labels}\n")
            exit()
        result_dict = {
            'phrase_acc': phrase_acc,
            'token_acc': token_acc,
            'token_norm': token_norm,
            'doc_norm': doc_norm,
            'phrase_norm': phrase_norm,
            'query_norm': query_norm_mean,
            'token_IP': token_IP,
            'phrase_IP': phrase_IP,
            'token_cos': token_cos,
            'phrase_cos': phrase_cos,
            'wrong_phrase_to_token': wrong_phrase_to_token,
            'phrase_inner_acc': phrase_inner_acc,
            'token_inner_acc': token_inner_acc
        }
        self.update_prebatch_phrase()
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