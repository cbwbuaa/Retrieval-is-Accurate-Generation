from header import *
# import spacy
import random, collections
from .util_func import *
from time import time
class Agent:
    
    def __init__(self, model, args):
        super(Agent, self).__init__()
        self.args = args
        self.model = model
        if 'resume' in args:
            self.load_last_step = args["resume"]
        else:
            self.load_last_step = False

        if torch.cuda.is_available():
            self.model.cuda()
        if args['mode'] in ['train', 'train_asyn', 'train_pipeline', 'pretrain', 'queryside', 'baseline']:
            self.set_optimizer_scheduler_ddp()
        if args['model'] == 'gpt2':
            self.train_model = self.train_model_gpt2
        if self.load_last_step:
            self.load_latest_checkpoint()
        else:
            if 'pretrain_model_path' in args and args["pretrain_model_path"] is not None:
                self.load_pretrain_model(args["pretrain_model_path"])
            if 'trained_model_path' in args and args["trained_model_path"] is not None:
                self.load_trained_model(args["trained_model_path"])
        self.result = collections.defaultdict(int)

    def evaluate_model(self, current_step, quiet=True):
        self.model.eval()
        print('[!] evaluating step', current_step)
        all_result, tok_counter, phrase_counter = self.model.module.evaluate(quiet=quiet)
        with open(f'{self.args["log_dir"]}/{self.args["mode"]}/{self.args["version"]}.log', 'a') as fLog:
            fLog.write(f'step: {current_step}\n')
            for k, v in all_result.items():
                if 'token' in k:
                    fLog.write('%s: %.4f\n' % (k, v / tok_counter))
                elif 'phrase' in k:
                    fLog.write('%s: %.4f\n' % (k, v / phrase_counter))
                elif 'global' in k:
                    fLog.write('%s: %.4f\n' % (k, v / (tok_counter + phrase_counter)))

    def set_optimizer_scheduler_ddp(self):
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        )
        self.scaler = GradScaler()
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=self.args['warmup_step'], 
            num_training_steps=self.args['total_step'],
        )
        find_unused_parameters = False #True if self.args['mode'] != 'queryside' else False
        self.model = nn.parallel.DistributedDataParallel(
            self.model, 
            device_ids=[self.args['local_rank']], 
            output_device=self.args['local_rank'],
            find_unused_parameters=find_unused_parameters,
        )

    def load_model(self, path):
        if self.args['mode'] in ['train', 'pretrain', 'train_asyn', 'train_pipeline']:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            model_state_dict = state_dict['model_state_dict']
            self.model.module.load_state_dict(model_state_dict)
            self.load_last_step = state_dict['step']
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        else:
            strict = True #self.args['mode'] == 'queryside'
            state_dict = torch.load(path, map_location=torch.device('cpu'))['model_state_dict']
            # print([x[0] for x in self.model.named_parameters()])
            # print('*' * 10)
            # print(list(state_dict.keys()))
            try:
                self.model.module.load_state_dict(state_dict, strict=strict)
            except:
                self.model.load_state_dict(state_dict, strict=strict)
        print(f'[!] resume model from {path}')
    
    def load_pretrain_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model_state_dict = state_dict['model_state_dict']
        self.model.module.load_state_dict(model_state_dict, strict=True)
        print(f'[!] load pretrained model from {path}')
    
    def load_trained_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model_state_dict = state_dict['model_state_dict']
        self.model.module.load_state_dict(model_state_dict, strict=True)
        print(f'[!] load trained model from {path}')
    
    def _train_model(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()
        with autocast():
            batch['current_step'] = current_step
            loss_0, loss_1, loss_2, acc_0, phrase_start_acc, phrase_end_acc, token_start_acc, token_end_acc = self.model(batch)
            loss = loss_0 + loss_1 + loss_2
            loss = loss / self.args['iter_to_accumulate']
        self.scaler.scale(loss).backward()
        if (current_step + 1) % self.args['iter_to_accumulate'] == 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

        if recoder:
            self.total_tok_acc += acc_0
            self.total_phrase_acc += (phrase_start_acc + phrase_end_acc) / 2
            recoder.add_scalar(f'train/Loss', loss.item(), current_step)
            recoder.add_scalar(f'train/pure_token_head_loss', loss_0.item(), current_step)
            recoder.add_scalar(f'train/start_loss', loss_1.item(), current_step)
            recoder.add_scalar(f'train/end_loss', loss_2.item(), current_step)
            recoder.add_scalar(f'train/pure_token_acc', acc_0, current_step)
            recoder.add_scalar(f'train/token_start_acc', token_start_acc, current_step)
            recoder.add_scalar(f'train/token_end_acc', token_end_acc, current_step)
            recoder.add_scalar(f'train/phrase_start_acc', phrase_start_acc, current_step)
            recoder.add_scalar(f'train/phrase_end_acc', phrase_end_acc, current_step)
        pbar.set_description(f'[!] loss(s|e): {round(loss_1.item(), 4)}|{round(loss_2.item(), 4)}; acc: {round(acc_0, 4)}|{round((token_start_acc+token_end_acc)/2, 4)}|{round((phrase_start_acc+phrase_end_acc)/2, 4)}')
        pbar.update(1)

    def train_model(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        # self.optimizer.zero_grad()
        with autocast():
            batch['current_step'] = current_step
            loss, result_dict = self.model(batch)
            # print(loss)
            loss = loss / self.args['iter_to_accumulate']
        self.scaler.scale(loss).backward()
        if (current_step + 1) % self.args['iter_to_accumulate'] == 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        # if self.args['local_rank'] == 0:
        #     for k, v in result_dict.items():
        #         if v != -1:
        #             self.result[k] += v
        #             self.result[k + '_cnt'] += 1
        # if recoder:
        #     recoder.add_scalar(f'train/Loss', loss.item(), current_step)
        #     recoder.add_scalar(f'train/token_acc', token_acc, current_step)
        #     recoder.add_scalar(f'train/phrase_acc', phrase_acc, current_step)
        # pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; acc(t|p): {round(token_acc, 4)}|{round(phrase_acc, 4)}')
        # pbar.update(1)

    def pretrain_model(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()
        with autocast():
            batch['current_step'] = current_step
            s_loss, e_loss, s_acc, e_acc = self.model(batch)
            loss = s_loss + e_loss
            loss = loss / self.args['iter_to_accumulate']
        self.scaler.scale(loss).backward()
        if (current_step + 1) % self.args['iter_to_accumulate'] == 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

        if recoder:
            recoder.add_scalar(f'train/Loss', loss.item(), current_step)
            recoder.add_scalar(f'train/start_loss', s_loss.item(), current_step)
            recoder.add_scalar(f'train/end_loss', e_loss.item(), current_step)
            recoder.add_scalar(f'train/phrase_start_acc', s_acc.item(), current_step)
            recoder.add_scalar(f'train/phrase_end_acc', e_acc.item(), current_step)
        pbar.set_description(f'[!] loss(s|e): {round(s_loss.item(), 4)}|{round(e_loss.item(), 4)}; acc: {round(s_acc.item(), 4)}|{round(e_acc.item(), 4)}')
        pbar.update(1)
    
    def queryside_tuning_model(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()
        with autocast():
            batch['current_step'] = current_step
            loss_0, loss_1, acc_0, phrase_acc = self.model(batch)
            loss = loss_0 + loss_1
            loss = loss / self.args['iter_to_accumulate']
        self.scaler.scale(loss).backward()
        if (current_step + 1) % self.args['iter_to_accumulate'] == 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
        if recoder:
            self.total_tok_acc += acc_0
            self.total_phrase_acc += phrase_acc
            recoder.add_scalar(f'train/Loss', loss.item(), current_step)
            recoder.add_scalar(f'train/token_loss', loss_0.item(), current_step)
            recoder.add_scalar(f'train/phrase_loss', loss_1.item(), current_step)
            recoder.add_scalar(f'train/token_acc', acc_0, current_step)
            recoder.add_scalar(f'train/phrase_acc', phrase_acc, current_step)
        pbar.set_description(f'[!] loss(t|p): {round(loss_0.item(), 4)}|{round(loss_1.item(), 4)}; acc: {round(acc_0, 4)}|{round(phrase_acc, 4)}')
        pbar.update(1)

    def load_latest_checkpoint(self):
        path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/{self.args["mode"]}'
        prefix_name = f'best_{self.args["version"]}_'
        checkpoints = []
        for file in os.listdir(path):
            if prefix_name in file:
                version = file[len(prefix_name):].strip('.pt')
                version = int(version)
                checkpoints.append((file, version))
        if len(checkpoints) == 0:
            print(f'[!] do not find the latest model checkpoints')
            return
        checkpoints = sorted(checkpoints, key=lambda x:x[-1])
        latest_checkpoint, version = checkpoints[-1]
        latest_checkpoint = os.path.join(path, latest_checkpoint)
        self.load_model(latest_checkpoint)
        self.load_last_step = version
        print(f'[!] train start from step: {version}')

    def save_model_long(self, path, current_step):
        model_state_dict = self.model.module.state_dict()
        scheduler_state_dict = self.scheduler.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save(
            {
                'model_state_dict' : model_state_dict,
                'scheduler_state_dict': scheduler_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'step': current_step
            }, 
            path
        )
        print(f'[!] save model into {path}')
        # with open(f'{self.args["log_dir"]}/{self.args["mode"]}/{self.args["version"]}.log', 'a') as fLog:
        #     fLog.write(f'step: {current_step}, ')
        #     for k, v in self.result.items():
        #         if k.endswith('_cnt'):
        #             continue
        #         fLog.write(f'{k}: {round(v / self.result[k + "_cnt"], 4)}\n')

        # for k in self.result.keys():
        #     self.result[k] = 0

    @torch.no_grad()
    def generate_multiple_sample(self, text, retriever, decoding_method='greedy', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False, random_seeds=[], reference=None):
        '''generate multiple samples by using the same set of phrases with differnt random seed'''
        self.model.eval()
        assert decoding_method == 'nucleus_sampling'
        sample_num = len(random_seeds)
        documents = retriever.search([text], self.args['doc_topk'])[0]
        phrase_reps, phrase_sources = self.get_phrases_fast(documents)
        collections = {s: None for s in random_seeds}
            
        ids_original = self.model.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
        prefix_length = len(ids_original[0])
        for i in range(sample_num):
            ids = ids_original.clone()

            torch.manual_seed(random_seeds[i])
            torch.cuda.manual_seed_all(random_seeds[i])
            candidates = []
            encode_time = 0

            bt = time()
            while len(ids[0]) <= prefix_length + self.args['max_gen_len']:
                ids, candidate = self.generate_one_step_fast(ids, phrase_reps, phrase_sources, decoding_method=decoding_method, top_k=top_k, top_p=top_p, temp=temp)
                candidates.append(candidate)
                # encode the document prefix
                if len(ids[0]) > 32 and encode_time == 0:
                    prefix_phrase_reps, prefix_phrase_sources = self.get_prefix_phrases_fast([self.model.tokenizer.decode(ids[0])])
                    phrase_reps = torch.cat([phrase_reps, prefix_phrase_reps], dim=0)
                    phrase_sources.extend(prefix_phrase_sources)
                    encode_time += 1
            inference_time = time() - bt
            collections[random_seeds[i]] = {
                'prefix': text,
                'reference': reference,
                'text': self.model.tokenizer.decode(ids[0, prefix_length:]),
                'phrases': candidates
            }
            if get_time_cost:
                collections[random_seeds[i]]['time_cost'] = inference_time
        return collections

    # baseline generation method
    @torch.no_grad()
    def generate_one_sample(self, text, retriever, decoding_method='greedy', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False):
        self.model.eval()
        ids = self.model.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
        prefix_length = len(ids[0])
        documents = retriever.search([text], self.args['doc_topk'])[0]
        # add the prefix
        # documents = [text] + documents
        phrase_reps, phrase_sources = self.get_phrases_fast(documents)
        candidates = []
        encode_time = 0
        bt = time()
        while len(ids[0]) <= prefix_length + self.args['max_gen_len']:
            ids, candidate = self.generate_one_step_fast(ids, phrase_reps, phrase_sources, decoding_method=decoding_method, top_k=top_k, top_p=top_p, temp=temp)
            candidates.append(candidate)
            # encode the document prefix
            if len(ids[0]) > 32 and encode_time == 0:
                prefix_phrase_reps, prefix_phrase_sources = self.get_prefix_phrases_fast([self.model.tokenizer.decode(ids[0])])
                phrase_reps = torch.cat([phrase_reps, prefix_phrase_reps], dim=0)
                phrase_sources.extend(prefix_phrase_sources)
                encode_time += 1
        inference_time = time() - bt
        if get_time_cost:
            return self.model.tokenizer.decode(ids[0, prefix_length:]), candidates, inference_time
        else:
            return self.model.tokenizer.decode(ids[0, prefix_length:]), candidates, None

    @torch.no_grad()
    def generate_one_step_fast(self, ids, phrase_reps, phrase_sources, decoding_method='greedy', temp=1., top_k=0, top_p=0.92):
        self.model.eval()
        query = self.model.get_query_rep(ids)
        score = torch.matmul(query, phrase_reps.t()).squeeze(0)   

        if decoding_method == 'greedy':
            index = score.max(dim=-1)[1].item()
            candidate = phrase_sources[index]
        elif decoding_method == 'nucleus_sampling':
            score = top_k_top_p_filtering(score, top_k=top_k, top_p=top_p)
            index = torch.multinomial(F.softmax(score/temp, dim=-1), num_samples=1).item()
            candidate = phrase_sources[index]
        else:
            pass

        # get textual candidate
        if type(candidate) == list:
            candidate = ' ' + self.model.bert_tokenizer.decode(candidate).replace('[UNK]', '<|endoftext|>')
            sub_ids = self.model.tokenizer.encode(candidate, add_special_tokens=False)
        else:
            sub_ids = [candidate]
        sub_ids = torch.LongTensor(sub_ids).unsqueeze(0).cuda()
        ids = torch.cat((ids, sub_ids), dim=-1)
        return ids, candidate

    # retrieve docs where candidates also exist
    @torch.no_grad()
    def generate_from_candidate_docs(self, text, candidate_list, base_data, phrase2doc_map, max_doc_num=1024, decoding_method='greedy', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False):
        self.model.eval()
        ids = self.model.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
        prefix_length = len(ids[0])
        # documents = retriever.search([text], self.args['doc_topk'])[0]
        doc_per_tok = 50
        candidate_doc_idx = set()
        for phrase in candidate_list:
            if phrase in phrase2doc_map:
                new_candidates = phrase2doc_map[phrase].keys()
                if len(new_candidates) > doc_per_tok:
                    candidate_doc_idx |= set(random.sample(new_candidates, doc_per_tok))
                else:
                    candidate_doc_idx |= set(new_candidates)
        print(f'get {len(candidate_doc_idx)} candidate docs.')
        if len(candidate_doc_idx) > max_doc_num:
            candidate_doc_idx = set(random.sample(candidate_doc_idx, max_doc_num))
        candidate_docs = [base_data[x] for x in candidate_doc_idx]
        # add the prefix
        # documents = [text] + documents
        phrase_reps, phrase_sources = self.get_phrases_fast(candidate_docs)
        candidates = []
        encode_time = 0
        bt = time()
        while len(ids[0]) <= prefix_length + self.args['max_gen_len']:
            ids, candidate = self.generate_one_step_fast(ids, phrase_reps, phrase_sources, decoding_method=decoding_method, top_k=top_k, top_p=top_p, temp=temp)
            candidates.append(candidate)
            if len(candidate_doc_idx) < max_doc_num:
                if candidate in phrase2doc_map:
                    new_candidate_doc_idx = phrase2doc_map[phrase].keys()
                    todo = set()
                    for x in new_candidate_doc_idx:
                        if x not in candidate_doc_idx:
                            todo.add(x)
                    new_candidate_doc_idx = todo
                    if len(new_candidate_doc_idx) > doc_per_tok:
                        new_candidate_doc_idx = random.sample(new_candidate_doc_idx, doc_per_tok)
                    if len(candidate_doc_idx) + len(new_candidate_doc_idx) > max_doc_num:
                        new_candidate_doc_idx = random.sample(new_candidate_doc_idx, max_doc_num - len(candidate_doc_idx))
                    print(f'add {len(new_candidate_doc_idx)} candidate docs.')
                    new_phrase_reps, new_phrase_sources = self.get_phrases_fast([base_data[x] for x in new_candidate_doc_idx])
                    phrase_reps = torch.vstack(phrase_reps, new_phrase_reps)
                    phrase_sources.extend(new_phrase_sources)
            # encode the document prefix
            if len(ids[0]) > 32 and encode_time == 0:
                prefix_phrase_reps, prefix_phrase_sources = self.get_prefix_phrases_fast([self.model.tokenizer.decode(ids[0])])
                phrase_reps = torch.cat([phrase_reps, prefix_phrase_reps], dim=0)
                phrase_sources.extend(prefix_phrase_sources)
                encode_time += 1
        inference_time = time() - bt
        if get_time_cost:
            return self.model.tokenizer.decode(ids[0, prefix_length:]), candidates, inference_time
        else:
            return self.model.tokenizer.decode(ids[0, prefix_length:]), candidates, None

    # generate based on gt docs
    @torch.no_grad()
    def generate_test(self, text, candidate_docs, decoding_method='greedy', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False):
        self.model.eval()
        ids = self.model.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
        prefix_length = len(ids[0])
        if type(candidate_docs[0]) == str: # docs
            phrase_reps, phrase_sources = self.get_phrases_fast(candidate_docs)
        else: # list of phrases
            phrase_reps, phrase_sources = self.get_phrases_test_fast(candidate_doc_phrases)
        candidates = []
        bt = time()
        while len(ids[0]) <= prefix_length + self.args['max_gen_len']:
            ids, candidate = self.generate_one_step_fast(ids, phrase_reps, phrase_sources, decoding_method=decoding_method, top_k=top_k, top_p=top_p, temp=temp)
            candidates.append(candidate)

        inference_time = time() - bt
        if get_time_cost:
            return self.model.tokenizer.decode(ids[0, prefix_length:]), candidates, inference_time
        else:
            return self.model.tokenizer.decode(ids[0, prefix_length:]), candidates, None

    @torch.no_grad()
    def get_phrases_test_fast(self, phrases_list):
        self.model.eval()
        documents = [' '.join(x) for x in phrases_list]
        # feed the 1024 maybe to big, leading to OOM
        inner_batch_size = 256
        offset_mapping, begin_hidden_states, end_hidden_states, vl, doc_ids = [], [], [], [], []
        for idx in range(0, len(documents), inner_batch_size):
            s_index, e_index = idx, idx + inner_batch_size
            batch_doc = documents[s_index:e_index]
            batch = self.model.bert_tokenizer.batch_encode_plus(batch_doc, padding=True, return_tensors='pt', max_length=256, truncation=True, return_offsets_mapping=True)
            input_ids = batch['input_ids'].cuda()
            mask = batch['attention_mask'].cuda()
            offset_mapping.extend(batch['offset_mapping'])
            hs = self.model.phrase_encoder(input_ids, mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
            bhs = self.model.s_proj(hs)
            ehs = self.model.e_proj(hs)
            begin_hidden_states.extend(bhs)
            end_hidden_states.extend(ehs)
            vl.extend(mask.sum(dim=-1))
            doc_ids.extend(input_ids.tolist())
        assert len(end_hidden_states) == len(begin_hidden_states) == len(documents) == len(vl) == len(doc_ids)

        begin_rep, end_rep = [], []
        phrase_sources = []
        # remove duplication in the phrase tables
        for phrases, doc, begin_doc_rep, end_doc_rep, l, doc_id, offset in zip(phrases_list, documents, begin_hidden_states, end_hidden_states, vl, doc_ids, offset_mapping):
            s_pos, e_pos = [], []
            st_pos = [x[0] for x in offset[1: l-1]]
            end_pos = [x[1] for x in offset[1: l-1]]
            cur_pos = 0
            for phrase in phrases:
                cur_pos = doc.find(phrase, cur_pos)
                try:
                    st_idx = st_pos.index(cur_pos) + 1
                    end_idx = end_pos.index(cur_pos + len(phrase)) + 1
                    s_pos.append(st_idx)
                    e_pos.append(end_idx)
                    phrase_sources.append(doc_id[st_idx: end_idx + 1])
                except:
                    pass
                cur_pos += len(phrase)
            s_rep = begin_doc_rep[s_pos, :]
            e_rep = end_doc_rep[e_pos, :]
            begin_rep.append(s_rep)
            end_rep.append(e_rep)
        if not begin_rep:
            phrase_reps = self.model.token_embeddings
            phrase_sources = [idx for idx in range(len(self.model.tokenizer))]
        else:
            begin_rep = torch.cat(begin_rep)
            end_rep = torch.cat(end_rep)
            phrase_reps = torch.cat([begin_rep, end_rep], dim=-1)
            phrase_reps = torch.cat([
                phrase_reps,
                self.model.token_embeddings
            ], dim=0)
            phrase_sources.extend([idx for idx in range(len(self.model.tokenizer))])
        return phrase_reps, phrase_sources
    

    # for phrase retrieval
    @torch.no_grad()
    def retrieve_one_phrase(self, text, retriever, phrase_sources, decoding_method='greedy', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False):
        self.model.eval()
        ids = self.model.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
        prefix_length = len(ids[0])
        candidates = []
        bt = time()
        while len(ids[0]) <= prefix_length + self.args['max_gen_len']:
            ids, candidate = self.retrieve_one_step_fast(ids, retriever, phrase_sources, decoding_method=decoding_method, top_k=top_k, top_p=top_p, temp=temp)
            candidates.append(candidate)
        inference_time = time() - bt
        if get_time_cost:
            return self.model.tokenizer.decode(ids[0, prefix_length:]), candidates, inference_time
        else:
            return self.model.tokenizer.decode(ids[0, prefix_length:]), candidates, None

    @torch.no_grad()
    def retrieve_one_step_fast(self, ids, retriever, phrase_sources, end_token='<|endoftext|>', decoding_method='greedy', temp=1., top_k=0, top_p=0.92):
        self.model.eval()
        query = self.model.get_query_rep(ids)#.cpu() #.numpy()
        topk_phrase = 128
        D, I = retriever.search(query.cpu(), topk_phrase)
        # D = torch.from_numpy(D)
        if decoding_method == 'greedy':
            index = I[0][0].item()
        elif decoding_method == 'nucleus_sampling':
            score = top_k_top_p_filtering(D[0], top_k=top_k, top_p=top_p)
            index = torch.multinomial(F.softmax(score/temp, dim=-1), num_samples=1)
            index = I[0][index].item()
        else:
            raise NotImplementedError

        if index < self.model.vocab_size:
            candidate = index
        else:
            candidate = phrase_sources[index - self.model.vocab_size]
        # get textual candidate
        if type(candidate) == int: # tok
            # candidate = ' ' + self.model.bert_tokenizer.decode(candidate).replace('[UNK]', '<|endoftext|>')
            # sub_ids = self.model.tokenizer.encode(candidate, add_special_tokens=False)
            sub_ids = [candidate]
            candidate = self.model.tokenizer.convert_ids_to_tokens(candidate).replace('[UNK]', '<|endoftext|>')
        elif type(candidate) == str: # phrase
            candidate = ' ' + candidate
            sub_ids = self.model.tokenizer.encode(candidate, add_special_tokens=False)
        else:
            raise NotImplementedError
        sub_ids = torch.LongTensor(sub_ids).unsqueeze(0).cuda()
        ids = torch.cat((ids, sub_ids), dim=-1)
        return ids, candidate

    @torch.no_grad()
    def get_phrases_fast(self, documents, add_token=True):
        self.model.eval()

        # feed the 1024 maybe too big, leading to OOM
        inner_batch_size = 256
        begin_hidden_states, end_hidden_states, vl, doc_ids = [], [], [], []
        for idx in range(0, len(documents), inner_batch_size):
            s_index, e_index = idx, idx + inner_batch_size
            batch_doc = documents[s_index: e_index]
            batch = self.model.bert_tokenizer.batch_encode_plus(batch_doc, padding=True, return_tensors='pt', max_length=512, truncation=True)
            input_ids = batch['input_ids'].cuda()
            mask = batch['attention_mask'].cuda()
            hs = self.model.phrase_encoder(input_ids, mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
            bhs = self.model.s_proj(hs)
            ehs = self.model.e_proj(hs)
            begin_hidden_states.extend(bhs)
            end_hidden_states.extend(ehs)
            vl.extend(mask.sum(dim=-1))
            doc_ids.extend(input_ids.tolist())
        assert len(end_hidden_states) == len(begin_hidden_states) == len(documents) == len(vl) == len(doc_ids)
        begin_rep, end_rep = [], []
        phrase_sources = []
        phrase_sources_set = set()
        # remove duplication in the phrase tables
        for idx, (begin_doc_rep, end_doc_rep, l, doc_id) in enumerate(zip(begin_hidden_states, end_hidden_states, vl, doc_ids)):
            s_pos, e_pos = [], []
            # ignore the [CLS] token
            for i in range(1, l-self.args['left_window_size']-1):
                # ignore the [SEP] token
                for j in range(
                    min(i + self.args['left_window_size'], l-1), 
                    min(i + self.args['right_window_size'], l-1)
                ):
                    phrase = doc_id[i:j+1]
                    if tuple(phrase) not in phrase_sources_set:
                        s_pos.append(i)
                        e_pos.append(j)
                        phrase_sources.append(phrase)
                        phrase_sources_set.add(tuple(phrase))
            s_rep = begin_doc_rep[s_pos, :]
            e_rep = end_doc_rep[e_pos, :]
            begin_rep.append(s_rep)
            end_rep.append(e_rep)
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([begin_rep, end_rep], dim=-1)
        if add_token:
            phrase_reps = torch.cat([
                phrase_reps,
                self.model.token_embeddings
            ], dim=0)
            phrase_sources.extend([idx for idx in range(len(self.model.tokenizer))])
        return phrase_reps, phrase_sources
    
    @torch.no_grad()
    def get_prefix_phrases_fast(self, documents):
        self.model.eval()
        batch = self.model.bert_tokenizer.batch_encode_plus(documents, padding=True, return_tensors='pt', max_length=200, truncation=True)
        input_ids = batch['input_ids'].cuda()
        mask = batch['attention_mask'].cuda()

        # replace the [CLS] with [PREFIX] for the prefix text (document)
        if hasattr(self.model, 'prefix_token_id'):
            input_ids[0, 0] = self.model.prefix_token_id

        vl = mask.sum(dim=-1)
        hidden_states = self.model.phrase_encoder(input_ids, mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]

        begin_rep, end_rep = [], []
        phrase_sources = []
        input_ids = input_ids.tolist()
        for doc_rep, l, doc_id in zip(hidden_states, vl, input_ids):
            s_pos, e_pos = [], []
            for i in range(1, l-self.args['left_window_size']):
                for j in range(
                    min(i+self.args['left_window_size'], l-1), 
                    min(i+self.args['right_window_size'], l-1)
                ):
                    s_pos.append(i)
                    e_pos.append(j)
                    phrase_sources.append(doc_id[i:j+1])
            s_rep = doc_rep[s_pos, :]
            e_rep = doc_rep[e_pos, :]
            begin_rep.append(s_rep)
            end_rep.append(e_rep)
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([self.model.s_proj(begin_rep), self.model.e_proj(end_rep)], dim=-1)
        return phrase_reps, phrase_sources

    def train_model_gpt2(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()
        with autocast():
            batch['current_step'] = current_step
            loss, acc = self.model(batch)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        if self.args['global_rank'] == 0:
            self.result['acc'] += acc
            self.result['step'] += 1
            if current_step % self.args['save_every'] == 0:
                with open(f'{self.args["log_dir"]}/{self.args["mode"]}/{self.args["version"]}.log', 'a') as fLog:
                    fLog.write(f'step: {current_step}, acc: {round(self.result["acc"] / self.result["step"] * 100, 2)}\n')
                    self.result['acc'] = 0
                    self.result['step'] = 0

    @torch.no_grad()
    def gpt2_generation(self, prefix, decoding_method='nucleus_sampling', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False):
        # maximum 128 tokens
        input_ids = self.model.vocab.encode(prefix, add_special_tokens=False)
        input_ids = torch.LongTensor(input_ids).unsqueeze(dim=0).cuda()
        length = len(input_ids[0])
        use_cache = False if get_time_cost else True
        bt = time()
        if decoding_method == 'nucleus_sampling':
            output = self.model.model.generate(
                input_ids,
                do_sample=True,
                max_length=length+128,
                top_p=top_p,
                top_k=0,
                use_cache=use_cache
            )
        else:
            output = self.model.model.generate(
                input_ids,
                max_length=length+128,
                use_cache=use_cache
            )
        inference_time = time() - bt
        string = self.model.vocab.decode(output[0, length:])
        return string, inference_time

    @torch.no_grad()
    def knnlm_generation(self, prefix, decoding_method='nucleus_sampling', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False):
        # maximum 128 tokens
        input_ids = self.model.vocab.encode(prefix, add_special_tokens=False)
        input_ids = torch.LongTensor(input_ids).unsqueeze(dim=0).cuda()
        length = len(input_ids[0])
        bt = time()
        if decoding_method == 'nucleus_sampling':
            string = self.model.nucleus_sampling(
                input_ids,
                max_length=128,
                top_p=top_p,
            )
        elif decoding_method == 'greedy':
            string = self.model.greedy_search(
                input_ids,
                max_length=128,
            )
        return string, time() - bt

    @torch.no_grad()
    def inference_knnlm(self, inf_iter, size=500000):
        self.model.eval()
        embds, texts = [], []
        counter = 0
        for batch in tqdm(inf_iter):
            rep, target = self.model(batch)
            embds.append(rep)
            texts.extend(target)
            if len(texts) > size:
                embds = torch.cat(embds, dim=0).numpy()
                torch.save(
                    (embds, texts), 
                    f'{self.args["root_dir"]}/data/{self.args["dataset"]}_1024/knnlm/inference_{self.args["local_rank"]}_{counter}.pt'
                )
                counter += 1
                texts, embds = [], []
        if len(texts) > 0:
            embds = torch.cat(embds, dim=0).numpy()
            torch.save(
                (embds, texts), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}_1024/knnlm/inference_{self.args["local_rank"]}_{counter}.pt'
            )

    @torch.no_grad()
    def test_model_ppl(self, test_iter, max_counter=10000):
        ppls = []
        counter = 0
        pbar = tqdm(test_iter)
        for batch in pbar:
            ppl = self.model.calculate_ppl(batch)
            ppls.append(ppl)
            counter += 1
            if counter >= max_counter:
                break
            ppl = np.mean(ppls)
            pbar.set_description(f'[!] ppl: {round(ppl, 4)}')
        print('Perplexity:', round(ppl, 4))

    @torch.no_grad()
    def get_phrase_emb(self, phrase_lists, doc_labels, docs_map):
        docs_text = [docs_map[x] for x in doc_labels]
        self.model.eval()
        all_s_rep, all_e_rep, all_offsets = self.model.encode_doc_batch(docs_text)
        all_phrase_emb = []
        all_phrase = []
        all_phrase_pos = []
        for phrases, doc_idx, doc, s_rep, e_rep, offset in zip(phrase_lists, doc_labels, docs_text, all_s_rep, all_e_rep, all_offsets):
            st_pos = [pos[0] for pos in offset[1:]]
            end_pos = [pos[1] for pos in offset[1:]]
            phrase_st_idx = []
            phrase_end_idx = []
            for phrase, phrase_st_pos in phrases:
                phrase_end_pos = phrase_st_pos + len(phrase)
                try:
                    st_idx_ = st_pos.index(phrase_st_pos) + 1
                    end_idx_ = end_pos.index(phrase_end_pos) + 1
                except:
                    continue
                phrase_st_idx.append(st_idx_)
                phrase_end_idx.append(end_idx_)
                all_phrase.append(phrase)
                all_phrase_pos.append((doc_idx, phrase_st_pos, phrase_end_pos))

            phrase_emb = torch.hstack((s_rep[torch.LongTensor(phrase_st_idx)], e_rep[torch.LongTensor(phrase_end_idx)]))
            if self.model.dim_proj is not None:
                phrase_emb = self.model.dim_proj(phrase_emb)
            all_phrase_emb.append(phrase_emb.cpu())

        return all_phrase, all_phrase_emb, all_phrase_pos
    
    # for QA test
    @torch.no_grad()
    def test_MRCQA(self, story, questions, answers):
        self.model.eval()
        questions_batch = self.model.tokenizer.batch_encode_plus(questions, padding=False, max_length=1024, truncation=True)
        ids = questions_batch['input_ids']
        tail_idx = [len(x) - 1 for x in ids]
        gpt2_ids = pad_sequence([torch.LongTensor(i) for i in ids], padding_value=self.model.tokenizer.eos_token_id, batch_first=True).cuda()
        gpt2_mask = generate_mask(gpt2_ids, pad_token_idx=self.model.tokenizer.eos_token_id).cuda()

        questions_reps_ = \
        self.model.model(input_ids=gpt2_ids, attention_mask=gpt2_mask, output_hidden_states=True).hidden_states[-1]
        questions_reps = questions_reps_[range(len(questions)), tail_idx]

        phrase_reps, phrase_sources = self.get_phrases_fast([story], add_token=False)
        score = torch.matmul(questions_reps, phrase_reps.t())
        preds = torch.argmax(score, dim=-1).view(-1).cpu().tolist()
        preds = [phrase_sources[x] for x in preds]
        preds = [self.model.bert_tokenizer.decode(x) for x in preds]
        return preds


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



