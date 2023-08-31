from header import *
from .util_func import *
from time import time
import random, collections
from tqdm import tqdm
from copy import deepcopy

class CopyisallyouneedPrefixOnly(Dataset):
    
    def __init__(self, **args):
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(args['prefix_encoder_tokenizer'][args['lang']])

        self.data_root_path = args['data_root_dir']
        self.phrase_embedding = np.memmap('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/phrase_embedding_index/PCA_emb_merged.npy', dtype=np.float32, mode='r', shape=(138543105, 128))
        self.file_lists = [f'{args["training_data_dir"]}/train_chunk{args["local_rank"]}']
        if args['local_rank'] == 0:
            print(f'[!] file list for worker{self.args["local_rank"]}: {self.file_lists}')
        
        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.buffer_size = args['buffer_size']
        self.last_delta = 0
        self.cache_prefix = []
        self.max_input_length = args['max_len']
        self.max_suffix_length = 128
        self.batch_size = args['batch_size']

        self.cache = []
        # self.data_queue = []
        # self.data_queue_size = 10000 # perhaps 1% overlap when 5000
        self.update_step = 0
        self._init_data_queue()
        if self.args['local_rank'] == 0:
            print('[!] Init data queue over')

    def __len__(self):
        return 10000000

    def _init_data_queue(self):
        self.load_one_part()
        # self.process_one_part(quiet=False, desc=f'Worker{self.args["local_rank"]} initializing data queue')
        # random.shuffle(self.data_queue)

    def load_one_part(self):
        self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        if len(self.cache) < self.buffer_size:
            # current file runs over, cyclely loading
            self.current_file_index = 0 if self.current_file_index == len(self.file_lists) - 1 else self.current_file_index + 1
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache += load_lines_chunk(self.current_file_handler, self.buffer_size - len(self.cache))
            with open(f'{args["log_dir"]}/{args["mode"]}/{args["version"]}.error', 'a') as f:
                f.write(f'\n{self.args["local_rank"]} finish epoch.\n')
        # self.cache = [json.loads(x) for x in self.cache]
        random.shuffle(self.cache)

    def _process_one_part(self, quiet=True, desc=''):
        stride = self.max_input_length
        for line in tqdm(self.cache, desc=desc, disable=quiet):
            document = line['doc']
            phrase_ref = line['phrase']
            tok_result = self.vocab(document, return_attention_mask=False, return_offsets_mapping=True)
            doc_toks = tok_result['input_ids']
            offset_mapping = tok_result['offset_mapping']
            start_map = [x[0] for x in offset_mapping]
            end_map = [x[1] for x in offset_mapping]
            for st in range(0, len(doc_toks), stride):
                end = min(st + self.max_input_length, len(doc_toks))
                gpt_ids = doc_toks[st: end]
                if len(gpt_ids) < 128:
                    continue
                if st == 0:
                    AR_start = 1
                else:
                    AR_start = max(1, self.max_input_length - stride)
                if AR_start >= len(gpt_ids): # already processed
                    break
                AR_mask = [0 for _ in range(AR_start)] + [1 for _ in range(len(gpt_ids) - AR_start)]
                valid_phrases = []
                for phrase, phrase_pos, ref_pos in phrase_ref:
                    start_idx, end_idx = None, None
                    phrase_st, phrase_end = phrase_pos
                    if phrase_st == 0:
                        continue
                    if phrase_st in start_map:
                        start_idx = start_map.index(phrase_st)
                    elif phrase_st - 1 in start_map:
                        start_idx = start_map.index(phrase_st - 1)
                        if end_map[start_idx] <= phrase_st:
                            start_idx = None
                    else:
                        start_idx = None
                    if start_idx:
                        if start_idx < st + AR_start:
                            continue
                        if phrase_end in end_map:
                            end_idx = end_map.index(phrase_end)
                        else:
                            end_idx = None
                    else:
                        continue
                    if start_idx and end_idx:
                        if start_idx >= st + AR_start and end_idx < end:
                            valid_phrases.append([phrase, start_idx - st, end_idx - st, ref_pos, document[phrase_st: phrase_st + self.max_suffix_length]])
                        elif end_idx >= end:
                            break
                self.data_queue.append([gpt_ids, valid_phrases, AR_mask])

    def _load_one_batch(self):
        all_gpt_ids, all_valid_phrases, all_AR_mask = [], [], []
        for _ in range(self.batch_size):
            if not self.data_queue:
                self.load_one_part()
                self.process_one_part()
                random.shuffle(self.data_queue)
            gpt_ids, valid_phrases, AR_mask = self.data_queue.pop(0)
            all_gpt_ids.append(gpt_ids)
            all_valid_phrases.append(valid_phrases)
            all_AR_mask.append(AR_mask)
        return all_gpt_ids, all_valid_phrases, all_AR_mask

    def load_one_batch(self):
        all_gpt_ids, all_valid_phrases, all_AR_mask = [], [], []
        for _ in range(self.batch_size):
            if not self.cache:
                self.load_one_part()
            gpt_ids, valid_phrases, AR_mask = json.loads(self.cache.pop(0))
            all_gpt_ids.append(gpt_ids)
            all_valid_phrases.append(valid_phrases)
            all_AR_mask.append(AR_mask)
        return all_gpt_ids, all_valid_phrases, all_AR_mask

    def __getitem__(self, i):
        data_st_time = time()
        batch = self.load_one_batch()
        all_gpt_ids, all_valid_phrases, all_AR_mask = batch

        # load phrase embeddings
        # keep the same phrases (relect popularities)
        embeddings = []
        phrase_list = []
        emb_idx_map = {}
        emb_idx_list = []
        for instance_idx, valid_phrases in enumerate(all_valid_phrases):
            for phrase, start_idx, end_idx, ref_pos, suffix in valid_phrases:
                emb_idx_map[(instance_idx, phrase, start_idx, end_idx)] = len(emb_idx_list)
                emb_idx_list.append(ref_pos)
                phrase_list.append(phrase)
        if emb_idx_list:
            embeddings = torch.from_numpy(self.phrase_embedding[emb_idx_list])
        else:
            embeddings = None
        t1 = time()

        # process the gpt2_batch
        all_gpt_ids = pad_sequence([torch.LongTensor(x) for x in all_gpt_ids], padding_value=self.vocab.eos_token_id, batch_first=True)
        all_AR_mask = pad_sequence([torch.LongTensor(x) for x in all_AR_mask], padding_value=self.vocab.eos_token_id, batch_first=True)
        all_gpt2_mask = generate_mask(all_gpt_ids, pad_token_idx=self.vocab.eos_token_id)
        # all_phrase_mask = deepcopy(all_AR_mask)
        all_phrase_label = deepcopy(all_gpt_ids)

        # all_phrase_mask, all_phrase_label = [], []
        all_false_neg_mask = []
        seq_len = all_gpt_ids.shape[1]
        # print('seq_len:', seq_len)
        for instance_idx, valid_phrases in enumerate(all_valid_phrases):
            # phrase_mask_ = [0 for _ in range(seq_len)]
            for phrase, start_idx, end_idx, ref_pos, suffix in valid_phrases:
                # # gpt mask
                # phrase_mask_[start_idx - 1] = 1
                # label
                emb_idx = emb_idx_map[(instance_idx, phrase, start_idx, end_idx)]
                # all_phrase_label.append(len(self.vocab) + emb_idx)
                all_phrase_label[instance_idx][start_idx] = len(self.vocab) + emb_idx
                # false neg
                false_neg_mask_ = [1 for _ in range(len(self.vocab) + len(phrase_list))]
                suffix += ' '
                for can_idx, candidate in enumerate(phrase_list):
                    if can_idx == emb_idx:
                        continue
                    candidate += ' '
                    if suffix.startswith(candidate):
                        false_neg_mask_[len(self.vocab) + can_idx] = 0
                all_false_neg_mask.append(false_neg_mask_)
            # all_phrase_mask.append(phrase_mask_)

        # prepare the batch
        all_false_neg_mask = torch.LongTensor(all_false_neg_mask)
        # all_phrase_label = torch.LongTensor(all_phrase_label)
        # all_phrase_mask = torch.LongTensor(all_phrase_mask)
        end_time = time()
        # print(f"data worker{self.args['local_rank']} time:", end_time - data_st_time)
        # print(all_gpt_ids.shape)
        # print(all_AR_mask.shape)
        # print(all_gpt2_mask.shape)
        # print(all_false_neg_mask.shape)
        # print(all_phrase_label)
        # print(all_phrase_mask.shape)
        # print(embeddings.shape)
        return all_gpt_ids, all_AR_mask, all_gpt2_mask, all_false_neg_mask, all_phrase_label, embeddings

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        all_gpt_ids, all_AR_mask, all_gpt2_mask, all_false_neg_mask, all_phrase_label, embeddings = batch[0]
        return {
            'gpt2_ids': all_gpt_ids,
            'gpt2_mask': all_gpt2_mask,
            'AR_mask': all_AR_mask,
            'false_neg_mask': all_false_neg_mask,
            'phrase_label': all_phrase_label,
            # 'phrase_mask': all_phrase_mask,
            'phrase_embedding': embeddings
        }

class PrebatchNegativePhrasesProcessor:
    def __init__(self, **args):
        self.prebatch_step_limit = args["prebatch_step"]
        self.prebatch_num = args["prebatch_num"]
        self.prebatch_phrases = []
    
    def _update_prebatch_phrase(self, cur_phrase_str):
        if self.prebatch_step_limit > 0:
            self.prebatch_phrases.append(cur_phrase_str)
            self.prebatch_phrases = self.prebatch_phrases[-self.prebatch_step_limit:]

    def _get_random_prebatch_phrase(self):       
        num_per_batch = self.prebatch_num
        if self.prebatch_phrases:
            all_str = []
            all_phrase_idx = []
            cur_pos = 0
            for phrase_str in self.prebatch_phrases:
                if num_per_batch == -1: # all
                    all_str.extend(phrase_str)
                    all_phrase_idx.extend(list(range(cur_pos, cur_pos + len(phrase_str))))
                else:
                    random_idx = random.sample(list(range(len(phrase_str))), num_per_batch)
                    all_str.extend([phrase_str[idx] for idx in random_idx])
                    all_phrase_idx.extend([cur_pos + x for x in random_idx])
                cur_pos += len(phrase_str)
            return all_str, all_phrase_idx
        else:
            return None, None
            
    def __call__(self, batch):
        # time1 = time()
        prebatch_str, prebatch_idx_batch = self._get_random_prebatch_phrase()
        if prebatch_str is not None:
            false_neg_mask = batch['false_neg_mask']
            false_neg_mask = false_neg_mask.reshape(-1, false_neg_mask.shape[-1])
            valid_tok_suffix = batch['valid_tok_suffix']
            prebatch_false_neg_mask = torch.ones(false_neg_mask.shape[0], len(prebatch_str))
            false_neg_idx0 = []
            false_neg_idx1 = []
            for suffix_idx, suffix in enumerate(valid_tok_suffix):
                if suffix:
                    suffix_ = suffix + ' '
                    for phrase_idx, phrase_ in enumerate(prebatch_str):
                        if suffix_.startswith(phrase_ + ' '):
                            false_neg_idx0.append(suffix_idx)
                            false_neg_idx1.append(phrase_idx)
            prebatch_false_neg_mask[false_neg_idx0, false_neg_idx1] = 0
            false_neg_mask = torch.cat([
                false_neg_mask,
                prebatch_false_neg_mask], dim=1)
            batch['false_neg_mask'] = false_neg_mask
        if prebatch_idx_batch is None:
            batch['prebatch_idx'] = None
        else:
            batch['prebatch_idx'] = torch.tensor(prebatch_idx_batch)
        self._update_prebatch_phrase(batch['all_phrases'])
        # print('process prebatch:', time() - time1)
        return batch
