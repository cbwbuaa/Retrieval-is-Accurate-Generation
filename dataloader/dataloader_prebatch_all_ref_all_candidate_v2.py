from header import *
from .util_func import *
from time import time
class CopyisallyouneedWikitext103V2DatasetPrebatchAllRefAllCandidateV2(Dataset):
    
    def __init__(self, **args):
        self.args = args
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_encoder_tokenizer'][args['lang']])
        self.bert_vocab.add_tokens(['<|endoftext|>'])
        self.vocab = AutoTokenizer.from_pretrained(args['prefix_encoder_tokenizer'][args['lang']])
        self.vocab.add_tokens(['<|endoftext|>'])

        self.data_root_path = args['data_root_dir']
        self.file_lists = [f'{args["training_data_dir"]}/tokenization_result_{i}.jsonl' for i in range(args['data_file_num'])]
        # count the number of the samples
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)

        if self.args['mode'] == 'train':
            new_seed = args['seed'] + args['global_rank']
            random.seed(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker {self.args["local_rank"]}:')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.last_delta = 0
        self.cache_prefix = []
        self.prebatch_neg = []
        self.max_suffix_length = 70
        self.pre_batch_step_limit = args["prebatch_phrase_num"] if 'prebatch_phrase_num' in args else 5

        # load the base_data
        base_data = {}
        lineid2docidx = {}
        with open(f'{self.data_root_path}/base_data_128.txt') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip().split('\t')
                chunk = ' '.join(line[:-1])
                id_label = line[-1].strip()
                if id_label:
                    base_data[id_label] = chunk
                    lineid2docidx[i] = id_label
        self.base_data = base_data
        if self.args['local_rank'] == 0:
            print(f'[!] load base data over')

        # load phrase candidates
        candidate_pos_map = {}
        line_id = 0
        for i in range(8):
            with open(f'/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_candidates/candidates_{i}.jsonl') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    pos = []
                    for phrase, st in data:
                        pos.append((st, st + len(phrase)))
                    doc_idx = lineid2docidx[line_id]
                    candidate_pos_map[doc_idx] = pos
                    line_id += 1
        self.candidate_pos_map = candidate_pos_map
        if self.args['local_rank'] == 0:
            print(f'[!] load candidates for {len(self.candidate_pos_map)} docs over')

    def __len__(self):
        return self.size

    def load_one_part(self):
        assert len(self.cache) == 0
        self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        if len(self.cache) == 0:
            # current file runs over, cyclely loading
            self.current_file_index = 0 if self.current_file_index == len(self.file_lists) - 1 else self.current_file_index + 1
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        self.cache = [json.loads(x) for x in self.cache]
        random.shuffle(self.cache)

    def load_one_batch(self):
        cache_doc, docs = set(), []
        prefix_batch, suffix_batch, gpt2_batch = [], [], []
        while len(cache_doc) < self.args['max_doc_size']:
            if len(self.cache) == 0:
                self.load_one_part()
            item = self.cache[0]
            if self.last_delta == len(item['results']):
                self.cache_prefix = []
                self.cache.pop(0)
                self.last_delta = 0
                continue
            phrase_list = []
            suffix_list = []            
            prefix_batch.append(self.cache_prefix)
            self.cache_prefix = []
            base_index = item['index']
            while self.last_delta < len(item['results']):
                phrase, metadata = item['results'][self.last_delta]
                if metadata and self.last_delta > 0:
                    doc, start_pos, end_pos = metadata
                    docs.append((doc, phrase, start_pos, end_pos))
                    cache_doc.add(doc)
                    phrase_list.append((phrase, 1))
                    # valid phrase add 1
                else:
                    phrase_list.append((phrase, 0))
                suffix_list.append(' '.join([x[0] for x in item['results'][self.last_delta: self.last_delta + self.max_suffix_length]]))
                
                self.last_delta += 1
                if len(cache_doc) >= self.args['max_doc_size']:
                    break
            
            if len(cache_doc) < self.args['max_doc_size']:
                self.cache.pop(0)
                self.last_delta = 0
                self.cache_prefix = []
            
            gpt2_batch.append(phrase_list)
            suffix_batch.append(suffix_list)
        return docs, prefix_batch, gpt2_batch, suffix_batch
    
    def update_prebatch_phrase(self, phrases):
        self.prebatch_neg.append(phrases)
        self.prebatch_neg = self.prebatch_neg[-self.pre_batch_step_limit:]
    
    def get_prebatch_neg(self):
        prebatch_neg = []
        for neg in self.prebatch_neg:
            prebatch_neg += neg
        return prebatch_neg

    def __getitem__(self, i):
        '''
        gpt2_batch: [B_v, S_v]
        bert_batch: [B_doc, S_doc]
        phrase_to_doc: [B_p]
        start_index: [B_p]
        end_index: [B_p]
        '''
        data_st_time = time()
        docs, prefix_batch, gpt2_batch, suffix_batch = self.load_one_batch()

        # process the bert_batch
        bert_batch, phrase_to_doc, phrase_start_index, phrase_end_index = [], [], [], []
        error_label, phrase_doc_dict = [], {}
        candidate_neg_pos = []
        target_phrases_str = []
        target_phrases_idx = [] # to remove from neg candidates
        for doc_id, phrase, start_pos, end_pos in docs:
            text = self.base_data[doc_id]
            item = self.bert_vocab(text, add_special_tokens=False, return_offsets_mapping=True)
            doc_ids = item['input_ids']
            start_mapping = [s for s, e in item['offset_mapping']]
            end_mapping = [e for s, e in item['offset_mapping']]
            try:
                # +1 cause the [CLS] token is added
                start_index = start_mapping.index(start_pos) + 1
                end_index = end_mapping.index(end_pos) + 1
            except:
                # wrong phrase, not consider
                error_label.append(True)
                continue
            if doc_id in phrase_doc_dict:
                phrase_to_doc.append(phrase_doc_dict[doc_id])
                target_phrases_idx[phrase_doc_dict[doc_id]].append((start_pos, end_pos))
            else:
                bert_batch.append(
                    [self.bert_vocab.cls_token_id] + doc_ids + [self.bert_vocab.sep_token_id]
                )
                phrase_to_doc.append(len(bert_batch) - 1)
                phrase_doc_dict[doc_id] = len(bert_batch) - 1
                # in doc neg
                candidate_neg_pos.append([])
                target_phrases_idx.append([(start_pos, end_pos)])
                if doc_id in self.candidate_pos_map:
                    candidates_pos = self.candidate_pos_map[doc_id]
                    for st, end in candidates_pos:
                        cand_text = text[st: end]
                        try:
                            st_idx = start_mapping.index(st) + 1
                            end_idx = end_mapping.index(end) + 1
                        except:
                            continue
                        candidate_neg_pos[-1].append((st_idx, end_idx, cand_text))
                else:
                    print('***', text, '***, do not have candidates')
            target_phrase = text[start_pos: end_pos]
            target_phrases_str.append(target_phrase)
            phrase_start_index.append(start_index)
            phrase_end_index.append(end_index)
            error_label.append(False)
        max_bert_length = max([len(i) for i in bert_batch]) if docs else None

        in_doc_neg_start_labels = []
        in_doc_neg_end_labels = []
        neg_phrases_str = []
        for idx, (target_pos, neg_pos) in enumerate(zip(target_phrases_idx, candidate_neg_pos)):
            for st, end, cand_text in neg_pos:
                if (st, end) in target_pos:
                    continue
                in_doc_neg_start_labels.append(st + max_bert_length * idx)
                in_doc_neg_end_labels.append(end + max_bert_length * idx)
                neg_phrases_str.append(cand_text)

        # process the gpt2_batch
        gpt2_ids, counter, valid_counter = [], 0, 0
        target_start_labels, target_end_labels, phrase_labels = [], [], []
        false_neg_mask = []
        all_phrases = target_phrases_str + neg_phrases_str + self.get_prebatch_neg()
        for batch_idx, (prefix, text, suffix) in enumerate(zip(prefix_batch, gpt2_batch, suffix_batch)):
            phrases = [phrase for phrase, _ in text]
            is_phrase = [label for _, label in text]
            gpt2_ids_ = []
            start_labels_ = []
            end_labels_ = []
            phrase_labels_ = []
            false_neg_mask_ = []
            if len(prefix) > 0:
                prev_token = prefix[-1].item()
                gpt2_ids_.append(prev_token)
                phrase_labels_.append(prev_token)
                false_neg_mask_.append(torch.ones(1, len(self.vocab) + len(all_phrases)))
            for idx, (phrase, label, phrase_suffix) in enumerate(zip(phrases, is_phrase, suffix)):
                if idx == 0 and len(prefix) == 0:
                    phrase_ = phrase
                else:
                    phrase_ = ' ' + phrase
                ids_ = self.vocab(phrase_, add_special_tokens=False)['input_ids']
                
                labels_ = deepcopy(ids_)
                mask_ = torch.ones(len(ids_), len(self.vocab) + len(all_phrases))
                false_neg_idx_ = []
                for phrase_idx, neg in enumerate(all_phrases):
                    if phrase_suffix.startswith(neg):
                        false_neg_idx_.append(phrase_idx + len(self.vocab))
                mask_[0][false_neg_idx_] = 0
                if label:
                    if error_label[counter] is False:
                        # replace the first token with OOV token to label it
                        bert_doc_id = phrase_to_doc[valid_counter]
                        chunk_length = max_bert_length * bert_doc_id
                        chunk_start_delta = phrase_start_index[valid_counter]
                        chunk_end_delta = phrase_end_index[valid_counter]
                        start_labels_.append(len(self.vocab) + chunk_length + chunk_start_delta)
                        end_labels_.append(len(self.vocab) + chunk_length + chunk_end_delta)
                        labels_[0] = len(self.vocab) + valid_counter
                        mask_[0][ids_[0]] = 0
                        mask_[0][len(self.vocab) + valid_counter] = 1
                        valid_counter += 1
                    counter += 1
                gpt2_ids_.extend(ids_)
                phrase_labels_.extend(labels_)
                false_neg_mask_.extend(mask_)
            gpt2_ids_ = torch.LongTensor(gpt2_ids_)
            phrase_labels_ = torch.LongTensor(phrase_labels_)
            false_neg_mask_ = torch.vstack(false_neg_mask_)
            
            valid_mask = gpt2_ids_ != self.vocab.eos_token_id
            valid_mask[0] = False
            phrase_labels_  = phrase_labels_[valid_mask]
            false_neg_mask_ = false_neg_mask_[valid_mask]
            # print(self.vocab.decode(gpt2_ids_))
            # print(gpt2_ids_)
            
            if len(prefix) > 1:
                # make sure modify prefix mask later
                gpt2_ids_ = torch.hstack((prefix[:-1], gpt2_ids_))
        
            gpt2_ids.append(gpt2_ids_)
            target_start_labels.extend(start_labels_)
            target_end_labels.extend(end_labels_)
            phrase_labels.append(phrase_labels_)
            false_neg_mask.append(false_neg_mask_)
        self.cache_prefix = gpt2_ids[-1]
        self.update_prebatch_phrase(target_phrases_str + neg_phrases_str)
        ######
        # prepare the batch
        false_neg_mask = torch.vstack(false_neg_mask)
        phrase_labels = torch.hstack(phrase_labels)
        gpt2_ids = pad_sequence(gpt2_ids, padding_value=self.vocab.eos_token_id, batch_first=True)
        target_start_labels = torch.LongTensor(target_start_labels)
        target_end_labels = torch.LongTensor(target_end_labels)

        gpt2_mask = generate_mask(gpt2_ids, pad_token_idx=self.vocab.eos_token_id)
        gpt2_label_mask = deepcopy(gpt2_mask)
        for i, prefix in enumerate(prefix_batch):
            if len(prefix) > 0:
                gpt2_label_mask[i][:len(prefix)] = 0
        # print(gpt2_mask.sum(), 'gpt2_mask:', gpt2_mask)
        # print(gpt2_label_mask.sum(), 'gpt2_label_mask:', gpt2_label_mask)
        # print('gpt2_ids:', gpt2_ids)
        # print('target gpt2_ids:', gpt2_ids[:, :-1].reshape(-1)[gpt2_label_mask[:, 1:].reshape(-1).to(torch.bool)])
        # print('phrase_labels:', phrase_labels)
        # print('length:', len(false_neg_mask), gpt2_label_mask[:, 1:].reshape(-1).sum())

        in_doc_neg_start_labels = torch.LongTensor(in_doc_neg_start_labels)
        in_doc_neg_end_labels = torch.LongTensor(in_doc_neg_end_labels)
        bert_ids = pad_sequence([torch.LongTensor(i) for i in bert_batch], padding_value=self.bert_vocab.pad_token_id, batch_first=True)
        bert_mask = generate_mask(bert_ids, pad_token_idx=self.bert_vocab.pad_token_id)
        return gpt2_ids, target_start_labels, target_end_labels, phrase_labels, bert_ids, gpt2_mask, gpt2_label_mask, bert_mask, in_doc_neg_start_labels, in_doc_neg_end_labels, false_neg_mask

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        gpt2_ids, target_start_labels, target_end_labels, phrase_labels, bert_ids, gpt2_mask, gpt2_label_mask, bert_mask, in_doc_neg_start_labels, in_doc_neg_end_labels, false_neg_mask = batch[0]
        return {
            'gpt2_ids': gpt2_ids.cuda(),
            'bert_ids': bert_ids.cuda(),
            'gpt2_mask': gpt2_mask.cuda(),
            'gpt2_label_mask': gpt2_label_mask.cuda(),
            'bert_mask': bert_mask.cuda(),
            'start_labels': target_start_labels.cuda(),
            'end_labels': target_end_labels.cuda(),
            'phrase_labels': phrase_labels.cuda(),
            'in_doc_neg_start_labels': in_doc_neg_start_labels.cuda(),
            'in_doc_neg_end_labels': in_doc_neg_end_labels.cuda(),
            'false_neg_mask': false_neg_mask.cuda()
        }
