from header import *
from .util_func import *


class CopyisallyouneedWikitext103V2DatasetPretrain(Dataset):
    
    def __init__(self, **args):
        self.args = args
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_encoder_tokenizer'][args['lang']])
        self.bert_vocab.add_tokens(['<|endoftext|>', '[PREFIX]'])
        self.prefix_token_id = self.bert_vocab.convert_tokens_to_ids('[PREFIX]')
        self.vocab = AutoTokenizer.from_pretrained(args['prefix_encoder_tokenizer'][args['lang']])
        self.data_root_path = args['data_root_dir']
        self.file_lists = [f'/apdcephfs/share_916081/ponybwcao/phrase_extraction/retrieve_doc/output/wikitext103/copyisallyouneed/ref_data/8split_0neg/pretrain_data_{i}.jsonl' for i in range(args['data_file_num'])]
        
        # count the number of the samples
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)

        if self.args['mode'] == 'pretrain':
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
        bert_batch: [B_doc, S_doc]
        start_index: [B_p]
        end_index: [B_p]
        '''
        docs, counter = [], 0
        while counter < self.args['max_doc_size']:
            if len(self.cache) == 0:
                self.load_one_part()
            item = json.loads(self.cache.pop(0).strip())
            base_index = item['index']
            phrases, valid_flags = item['results']
            counter += 1
            docs.append((phrases, valid_flags))

        # process the bert_batch
        bert_batch, phrase_start_index, phrase_end_index = [], [], []
        for phrases, valid_flags in docs:
            text = ' '.join(phrases)
            item = self.bert_vocab(text, add_special_tokens=False, return_offsets_mapping=True)
            doc_ids = item['input_ids']
            start_mapping = [s for s, e in item['offset_mapping']]
            end_mapping = [e for s, e in item['offset_mapping']]
            tok_st_idx = []
            tok_end_idx = []
            cur_pos = 0
            for phrase in phrases:
                try:
                    start_index = start_mapping.index(cur_pos)
                    end_index = end_mapping.index(cur_pos + len(phrase))
                    tok_st_idx.append(start_index + 1) # +1 for cls
                    tok_end_idx.append(end_index + 1)
                    cur_pos += len(phrase) + 1
                except:
                    cur_pos += len(phrase) + 1
                    # wrong phrase, not consider
                    continue
            
            bert_batch.append(
                [self.bert_vocab.cls_token_id] + doc_ids + [self.bert_vocab.sep_token_id]
            )
            phrase_start_index.append(torch.LongTensor(tok_st_idx))
            phrase_end_index.append(torch.LongTensor(tok_end_idx))

        ######
        # prepare the batch
        bert_ids = pad_sequence([torch.LongTensor(i) for i in bert_batch], padding_value=self.bert_vocab.pad_token_id, batch_first=True)
        bert_mask = generate_mask(bert_ids, pad_token_idx=self.bert_vocab.pad_token_id)

        start_labels = []
        end_labels = []
        for st_idx, end_idx, token_ids in zip(phrase_start_index, phrase_end_index, bert_batch):
            st_label = torch.zeros(bert_ids.shape[1]).scatter_(dim=0, index=st_idx, value=1)
            end_label = torch.zeros(bert_ids.shape[1]).scatter_(dim=0, index=end_idx, value=1)
            st_label[len(token_ids):] = -1
            end_label[len(token_ids):] = -1
            start_labels.append(st_label)
            end_labels.append(end_label)
        start_labels = torch.vstack(start_labels).long()
        end_labels = torch.vstack(end_labels).long()
        return start_labels, end_labels, bert_ids, bert_mask

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        start_labels, end_labels, bert_ids, bert_mask = batch[0]
        return {
            'bert_ids': bert_ids.cuda(),
            'bert_mask': bert_mask.cuda(),
            'start_labels': start_labels.cuda(),
            'end_labels': end_labels.cuda(),
        }