from header import *
from .util_func import *

class CopyisallyouneedWikitext103V2DatasetQueryside(Dataset):
    
    def __init__(self, **args):
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(args['prefix_encoder_tokenizer'][args['lang']])
        self.data_root_path = args['data_root_dir']
        self.file_lists = [f'{args["training_data_dir"]}/tokenization_result_{i}.jsonl' for i in range(args['data_file_num'])]
        # count the number of the samples
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)

        new_seed = args['seed'] + args['global_rank']
        random.seed(new_seed)
        random.shuffle(self.file_lists)
        print(f'[!] file list for worker {self.args["local_rank"]}:')
        print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0
        self.cur_pos = 0

        # load the base_data
        base_data = {}
        with open(f'{self.data_root_path}/base_data_128.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                chunk = ' '.join(line[:-1])
                id_label = line[-1].strip()
                if id_label:
                    base_data[id_label] = chunk
        self.base_data = base_data
        print(f'[!] load base data over')

        # load idx2emb map
        self.idx2emb = load_emb('/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/phrase_index/best_prebatch_neg0_pretrain40w_1000000/cluster_idx2emb_map.pkl')

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

    def __getitem__(self, i):
        '''
        gpt2_batch: [B_v, S_v]
        phrase_to_doc: [B_p]
        start_index: [B_p]
        end_index: [B_p]
        '''

        gpt2_batch, cache_doc, query_labels, counter = [], set(), [], 0
        while counter < self.args['max_query_num']:
            if len(self.cache) == 0:
                self.load_one_part()
            item = self.cache[0]
            base_index = item['index']
            cur_doc = self.base_data[base_index]
            cache_phrase, delta = [], 0
            for phrase, metadata in item['results'][self.last_delta:]:
                if metadata and delta > 0:
                    self.cur_pos = cur_doc.find(phrase, self.cur_pos)
                    phrase_ = ' ' + phrase
                    # doc, start_pos = metadata[0]
                    query_labels.append((base_index, self.cur_pos))
                    # save the phrase and its map to the cache_doc
                    cache_phrase.append((phrase_, 1))
                    # valid phrase add 1
                    counter += 1
                else: # executes after the loop completes normally
                    phrase_ = phrase
                    if delta > 0:
                        phrase_ = ' ' + phrase_
                    cache_phrase.append((phrase_, 0))
                self.cur_pos += len(phrase)
                if counter >= self.args['max_query_num']:
                    self.last_delta += delta
                    self.if_last_over = False
                    break
                delta += 1
            else:
                self.if_last_over = True
            # update the cache
            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)
                self.cur_pos = 0

            # collect
            gpt2_batch.append(cache_phrase)

        # process the gpt2_batch
        gpt2_ids, target_idxs = [], []
        valid_phrases = []
        valid_counter = 0
        for text in gpt2_batch:
            phrases = [phrase for phrase, _ in text]
            is_phrase = [label for _, label in text]
            phrase_ids = self.vocab(phrases, add_special_tokens=False)['input_ids']
            ids, target_idx = [], []
            for text, ids_, label in zip(phrases, phrase_ids, is_phrase):
                target_idx_ = deepcopy(ids_)
                if label:
                    if query_labels[valid_counter] in self.idx2emb:
                        target_idx_[0] = len(self.vocab) + self.idx2emb[query_labels[valid_counter]]
                        valid_phrases.append(text)
                    valid_counter += 1
                ids.extend(ids_)
                target_idx.extend(target_idx_)
            gpt2_ids.append(ids)
            target_idxs.append(target_idx)
            
        ######
        # prepare the batch
        gpt2_ids = pad_sequence([torch.LongTensor(i) for i in gpt2_ids], padding_value=self.vocab.eos_token_id, batch_first=True)
        gpt2_mask = generate_mask(gpt2_ids, pad_token_idx=self.vocab.eos_token_id)
        target_idxs = pad_sequence([torch.LongTensor(i) for i in target_idxs], padding_value=self.vocab.eos_token_id, batch_first=True)
        return gpt2_ids, gpt2_mask, target_idxs, valid_phrases

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        gpt2_ids, gpt2_mask, target_idxs, valid_phrases = batch[0]
        return {
            'gpt2_ids': gpt2_ids.cuda(),
            'gpt2_mask': gpt2_mask.cuda(),
            'target_idxs': target_idxs.cuda(),
            'phrases': valid_phrases
        }
