from header import *
from .util_func import *

class CopyisallyouneedWikitext103V2DatasetPrebatchAllRefAllCandidate(Dataset):
    
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

    def _truncate_triplet(self, a, b, c, max_length):
        while True:
            if len(a) + len(b) + len(c) <= max_length:
                break
            else:
                if len(a) > len(c):
                    a.pop(0)
                else:
                    c.pop()

    def load_one_batch(self):
        cache_doc, docs, cache_phrase = set(), [], []
        if len(self.cache) == 0:
            self.load_one_part()
        item = self.cache[0]
        base_index = item['index']
        # print('-----', self.base_data[base_index], '-----')
        while self.last_delta < len(item['results']):
            phrase, metadata = item['results'][self.last_delta]
            if metadata and self.last_delta > 0:
                doc, start_pos, end_pos = metadata
                truncate_length = None
                docs.append((doc, phrase, start_pos, end_pos, truncate_length))
                cache_doc.add(doc)
                cache_phrase.append((phrase, 1))
                # valid phrase add 1
            else:
                cache_phrase.append((phrase, 0))
            
            self.last_delta += 1
            if len(cache_doc) >= self.args['max_doc_size']:
                break
        # update the cache
        update_flag = False
        if self.last_delta == len(item['results']):
            self.last_delta = 0
            update_flag= True
            self.cache.pop(0)
        return cache_phrase, docs, update_flag

    def __getitem__(self, i):
        '''
        gpt2_batch: [B_v, S_v]
        bert_batch: [B_doc, S_doc]
        phrase_to_doc: [B_p]
        start_index: [B_p]
        end_index: [B_p]
        '''

        gpt2_batch = []

        cache_phrase, docs, update_flag = self.load_one_batch()
        while not docs:
            cache_phrase, docs, update_flag = self.load_one_batch()

        # collect
        gpt2_batch.append(cache_phrase)

        # process the bert_batch
        bert_batch, phrase_to_doc, phrase_start_index, phrase_end_index = [], [], [], []
        error_label, phrase_doc_dict, phrase_texts = [], {}, []
        candidate_neg_pos = []
        target_phrases = []
        for doc_id, phrase, start_pos, end_pos, truncate_length in docs:
            text = self.base_data[doc_id]
            phrase_texts.append(text[start_pos: end_pos])
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
            assert not truncate_length
            if doc_id in phrase_doc_dict:
                phrase_to_doc.append(phrase_doc_dict[doc_id])
                # make sure no candidate same as target phrase
                target_phrase = text[start_pos: end_pos]
                target_phrases.append(target_phrase)
                candidate_neg_pos[phrase_doc_dict[doc_id]] = [x for x in candidate_neg_pos[phrase_doc_dict[doc_id]] if x[0] != start_index]
            else:
                bert_batch.append(
                    [self.bert_vocab.cls_token_id] + doc_ids + [self.bert_vocab.sep_token_id]
                )
                phrase_to_doc.append(len(bert_batch) - 1)
                phrase_doc_dict[doc_id] = len(bert_batch) - 1
                # in doc neg
                candidate_neg_pos.append([])
                target_phrase = text[start_pos: end_pos]
                target_phrases.append(target_phrase)
                if doc_id in self.candidate_pos_map:
                    candidates_pos = self.candidate_pos_map[doc_id]
                    for st, end in candidates_pos:
                        cand_text = text[st: end]
                        # if cand_text == target_phrase:
                        #     continue
                        try:
                            st_idx = start_mapping.index(st) + 1
                            end_idx = end_mapping.index(end) + 1
                        except:
                            continue
                        # this is xx, this is a xx
                        if st_idx == start_index:
                            continue
                        candidate_neg_pos[-1].append((st_idx, end_idx, cand_text))
                else:
                    print('***', text, '***, do not have candidates')
            phrase_start_index.append(start_index)
            phrase_end_index.append(end_index)
            error_label.append(False)
        max_bert_length = max([len(i) for i in bert_batch]) if docs else None

        assert len(candidate_neg_pos) == len(bert_batch)
        in_doc_neg_start_labels = []
        in_doc_neg_end_labels = []
        for idx, neg_pos in enumerate(candidate_neg_pos):
            for st, end, cand_text in neg_pos:
                # make sure no candidate same as target phrase (as a prefix)
                valid_flag = True
                for target in target_phrases:
                    if target.startswith(cand_text):
                        valid_flag = False
                        break
                if valid_flag:
                    in_doc_neg_start_labels.append(st + max_bert_length * idx)
                    in_doc_neg_end_labels.append(end + max_bert_length * idx)

        # process the gpt2_batch
        gpt2_ids, counter, valid_counter = [], 0, 0
        start_labels, end_labels = [], []
        valid_phrases = []
        phrase_labels = [] # for tok and phrase
        phrase_label_counter = 0
        assert len(gpt2_batch) == 1
        for text in gpt2_batch:
            phrases = [phrase for phrase, _ in text]
            is_phrase = [label for _, label in text]
            gpt2_ids_ = []
            start_labels_ = []
            end_labels_ = []
            phrase_labels_ = []
            for idx, (phrase, label) in enumerate(zip(phrases, is_phrase)):
                beggining_flag = False
                if idx == 0 and len(self.cache_prefix) == 0:
                    phrase_ = phrase
                    beggining_flag = True
                else:
                    phrase_ = ' ' + phrase
                ids_ = self.vocab(phrase_, add_special_tokens=False)['input_ids']

                start_ids_ = deepcopy(ids_)
                end_ids_ = deepcopy(ids_)
                labels_ = deepcopy(ids_)
                if label and error_label[counter] is False:
                    # replace the first token with OOV token to label it
                    bert_doc_id = phrase_to_doc[valid_counter]
                    chunk_length = max_bert_length * bert_doc_id
                    chunk_start_delta = phrase_start_index[valid_counter]
                    chunk_end_delta = phrase_end_index[valid_counter]
                    if not beggining_flag:
                        start_ids_[0] = len(self.vocab) + chunk_length + chunk_start_delta
                        end_ids_[0] = len(self.vocab) + chunk_length + chunk_end_delta
                        labels_[0] = len(self.vocab) + phrase_label_counter
                        phrase_label_counter += 1
                        valid_phrases.append(phrase_texts[counter])
                    valid_counter += 1
                if beggining_flag:
                    start_ids_ = start_ids_[1:]
                    end_ids_ = end_ids_[1:]
                    labels_ = labels_[1:]
                if label:
                    counter += 1
                gpt2_ids_.extend(ids_)
                start_labels_.extend(start_ids_)
                end_labels_.extend(end_ids_)
                phrase_labels_.extend(labels_)
            # print(self.vocab.decode(gpt2_ids_))
            if len(gpt2_ids_) >= self.args['max_len']:
                print('Too long prefix:', self.vocab.decode(gpt2_ids_))
                gpt2_ids_ = gpt2_ids_[-self.args['max_len']:]
                start_labels_ = start_labels_[-self.args['max_len'] + 1:]
                end_labels_ = end_labels_[-self.args['max_len'] + 1:]
                phrase_labels_ = phrase_labels_[-self.args['max_len'] + 1:]
                valid_phrase_num = torch.tensor(phrase_labels_) >= len(self.vocab)
                valid_phrase_num = valid_phrase_num.sum().item()
                valid_phrases = valid_phrases[-valid_phrase_num:]
            else:
                gpt2_ids_ = self.cache_prefix + gpt2_ids_
                gpt2_ids_ = gpt2_ids_[-self.args['max_len']:]
            
            gpt2_ids.append(gpt2_ids_)
            start_labels.append(start_labels_)
            end_labels.append(end_labels_)
            phrase_labels.append(phrase_labels_)
            self.cache_prefix = gpt2_ids_

        if update_flag:
            self.cache_prefix = []

        ######
        # prepare the batch
        gpt2_ids = pad_sequence([torch.LongTensor(i) for i in gpt2_ids], padding_value=self.vocab.eos_token_id, batch_first=True)
        start_labels = pad_sequence([torch.LongTensor(i) for i in start_labels], padding_value=self.vocab.eos_token_id, batch_first=True)
        end_labels = pad_sequence([torch.LongTensor(i) for i in end_labels], padding_value=self.vocab.eos_token_id, batch_first=True)
        phrase_labels = pad_sequence([torch.LongTensor(i) for i in phrase_labels], padding_value=self.vocab.eos_token_id, batch_first=True)
        gpt2_mask = generate_mask(gpt2_ids, pad_token_idx=self.vocab.eos_token_id)
        if not docs:
            bert_ids, bert_mask = torch.tensor([]), torch.tensor([])
            in_doc_neg_start_labels, in_doc_neg_end_labels = torch.tensor([]), torch.tensor([])
        else:
            in_doc_neg_start_labels = pad_sequence([torch.LongTensor(in_doc_neg_start_labels)], padding_value=self.vocab.eos_token_id, batch_first=True)
            in_doc_neg_end_labels = pad_sequence([torch.LongTensor(in_doc_neg_end_labels)], padding_value=self.vocab.eos_token_id, batch_first=True)
            bert_ids = pad_sequence([torch.LongTensor(i) for i in bert_batch], padding_value=self.bert_vocab.pad_token_id, batch_first=True)
            bert_mask = generate_mask(bert_ids, pad_token_idx=self.bert_vocab.pad_token_id)

        return gpt2_ids, start_labels, end_labels, phrase_labels, bert_ids, gpt2_mask, bert_mask, valid_phrases, in_doc_neg_start_labels, in_doc_neg_end_labels

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ''' 0428 YQC edit: start'''
        gpt2_ids, start_labels, end_labels, phrase_labels, bert_ids, gpt2_mask, bert_mask, valid_phrases, in_doc_neg_start_labels, in_doc_neg_end_labels = batch[0]
        return {
            'gpt2_ids': gpt2_ids.cuda(),
            'bert_ids': bert_ids.cuda(),
            'gpt2_mask': gpt2_mask.cuda(),
            'bert_mask': bert_mask.cuda(),
            'start_labels': start_labels.cuda(),
            'end_labels': end_labels.cuda(),
            'phrase_labels': phrase_labels.cuda(),
            'in_doc_neg_start_labels': in_doc_neg_start_labels.cuda(),
            'in_doc_neg_end_labels': in_doc_neg_end_labels.cuda(),
            'phrases': valid_phrases
        }
        ''' 0428 YQC edit: end'''
