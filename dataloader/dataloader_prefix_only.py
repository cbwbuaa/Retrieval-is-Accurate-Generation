from header import *
from .util_func import *
from time import time
import random, collections
from tqdm import tqdm
from copy import deepcopy

class CopyisallyouneedPrefixOnly(Dataset):
    
    def __init__(self, **args):
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(args['prefix_encoder_tokenizer'][args['lang']][self.args['model_size']])

        # self.phrase_embedding = np.memmap('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/phrase_embedding_index/PCA_emb_merged_memmap.npy', dtype=np.float32, mode='r', shape=(137101097, 128))
        self.file_lists = [f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/minipile/minipile_wiki/all_merged_shuffled_chunk{args["local_rank"]}']
        # self.file_lists = [f'{args["training_data_dir"]}/train_chunk{args["local_rank"]}_tok_small_EM']
        if args['local_rank'] == 0:
            print(f'[!] file list for worker{self.args["local_rank"]}: {self.file_lists}')
        
        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.buffer_size = args['buffer_size']
        self.last_delta = 0
        self.cache_prefix = []
        self.max_input_length = args['max_len']
        self.batch_size = args['batch_size']
        if args['FN']:
            self.phrase_table = load_emb('/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/wikipedia/phrase_embedding_index/PCA_phrase_merged.pkl')

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
            if self.args['local_rank'] == 0:
                print(f'\n{self.args["local_rank"]} finish epoch.\n')
            # with open(f'{self.args["log_dir"]}/{self.args["mode"]}/{self.args["version"]}.error', 'a') as f:
            #     f.write(f'\n{self.args["local_rank"]} finish epoch.\n')
        # self.cache = [json.loads(x) for x in self.cache]
        random.shuffle(self.cache)

    def load_one_batch(self):
        all_gpt_ids, all_valid_phrases, all_document, all_global_suffix_st_char, all_hard_negs = [], [], [], [], []
        for _ in range(self.batch_size):
            if not self.cache:
                self.load_one_part()
            item = json.loads(self.cache.pop(0))
            if len(item) == 4:
                gpt_ids, valid_phrases, document, global_suffix_st_char = item
                hard_negs = []
            elif len(item) == 5:
                gpt_ids, valid_phrases, document, global_suffix_st_char, hard_negs = item
            all_gpt_ids.append(gpt_ids)
            all_valid_phrases.append(valid_phrases)
            all_document.append(document)
            all_global_suffix_st_char.append(global_suffix_st_char)
            all_hard_negs.append(hard_negs)
        return all_gpt_ids, all_valid_phrases, all_document, all_global_suffix_st_char, all_hard_negs

    def __getitem__(self, i):
        # data_st_time = time()
        batch = self.load_one_batch()
        all_gpt_ids, all_valid_phrases, all_document, all_global_suffix_st_char, all_hard_negs = batch

        # load phrase embeddings
        # keep the same phrases (reflect popularities)?
        embeddings = []
        phrase_list = []
        emb_idx_map = {}
        emb_idx_list = []
        for instance_idx in range(len(all_valid_phrases)):
            valid_phrases, hard_negs = all_valid_phrases[instance_idx], all_hard_negs[instance_idx]
            for phrase, start_idx, end_idx, ref_pos in valid_phrases:
                if ref_pos not in emb_idx_map:
                    emb_idx_map[ref_pos] = len(emb_idx_list)
                    emb_idx_list.append(ref_pos)
                    phrase_list.append(phrase)
            for neg_ref_pos_list in hard_negs:
                for neg_ref_pos in neg_ref_pos_list:
                    if neg_ref_pos not in emb_idx_map:
                        emb_idx_map[neg_ref_pos] = len(emb_idx_list)
                        emb_idx_list.append(neg_ref_pos)
                        if self.args['FN']:
                            phrase_list.append(self.phrase_table[neg_ref_pos])
        # t0 = time()
        # if emb_idx_list:
        #     embeddings = torch.from_numpy(self.phrase_embedding[emb_idx_list])
        # else:
        #     embeddings = None
        # t1 = time()

        # process the gpt2_batch
        all_gpt_ids = pad_sequence([torch.LongTensor(x) for x in all_gpt_ids], padding_value=self.vocab.eos_token_id, batch_first=True)
        all_gpt2_mask = generate_mask(all_gpt_ids, pad_token_idx=self.vocab.eos_token_id)
        all_phrase_label = deepcopy(all_gpt_ids)
        if self.args['FN']:
            all_false_neg_mask = torch.ones(*all_gpt_ids.shape, len(self.vocab) + len(phrase_list))
        else:
            all_false_neg_mask = None

        seq_len = all_gpt_ids.shape[1]
        max_suffix_length = 72
        for instance_idx, valid_phrases in enumerate(all_valid_phrases):
            document = all_document[instance_idx]
            global_suffix_st_char = all_global_suffix_st_char[instance_idx]
            if self.args['FN']:
                for tok_idx in range(len(global_suffix_st_char)):
                    suffix_st_char = global_suffix_st_char[tok_idx]
                    if suffix_st_char == -1: # inner token
                        continue
                    suffix_ = document[suffix_st_char: suffix_st_char + max_suffix_length].lstrip() + ' '
                    for can_idx, candidate in enumerate(phrase_list):
                        if suffix_.startswith(candidate + ' '):
                            all_false_neg_mask[instance_idx, tok_idx, len(self.vocab) + can_idx] = 0
            
            for phrase, start_idx, end_idx, ref_pos in valid_phrases:
                # label
                emb_idx = emb_idx_map[ref_pos]
                all_phrase_label[instance_idx][start_idx] = len(self.vocab) + emb_idx
                if self.args['FN']:
                    # false neg
                    all_false_neg_mask[instance_idx, start_idx, len(self.vocab) + emb_idx] = 1

        # prepare the batch
        if self.args['FN']:
            all_false_neg_mask = all_false_neg_mask.to(torch.bool)
        # end_time = time()
        # print(f"data worker{self.args['local_rank']} time:", end_time - data_st_time)
        return all_gpt_ids, all_gpt2_mask, all_false_neg_mask, all_phrase_label, emb_idx_list, phrase_list

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        all_gpt_ids, all_gpt2_mask, all_false_neg_mask, all_phrase_label, emb_idx_list, phrase_list = batch[0]
        return {
            'gpt2_ids': all_gpt_ids,
            'gpt2_mask': all_gpt2_mask,
            'false_neg_mask': all_false_neg_mask,
            'phrase_label': all_phrase_label,
            'phrase_embedding_idx_list': emb_idx_list,
            'phrase_list': phrase_list
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
