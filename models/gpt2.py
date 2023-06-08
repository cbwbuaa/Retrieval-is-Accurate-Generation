from header import *
import sys, random
import numpy as np
from time import time
from tqdm import tqdm
import collections
from .util_func import *
import faiss
import faiss.contrib.torch_utils

class GPT2Baseline(nn.Module):

    def __init__(self, **args):
        super(GPT2Baseline, self).__init__()
        model = args['pretrained_model'][args['lang']]
        self.model_name = model
        if args["random_initialize"]:
            print('Random Init.')
            GPT_config = AutoConfig.from_pretrained(model)
            self.model = GPT2LMHeadModel(GPT_config)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.vocab_size = len(self.tokenizer)
        self.args = args

        self.pad = self.tokenizer.eos_token_id
        self.unk = self.tokenizer.unk_token_id
        self.special_tokens = set([self.pad])
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)

    def forward(self, batch):
        ids, ids_mask = batch['ids'], batch['ids_mask']
        ids, ods = ids[:, :-1], ids[:, 1:]
        ids_mask = ids_mask[:, :-1]
        output = self.model(input_ids=ids, attention_mask=ids_mask)
        gen_logits = output.logits
        loss = self.gen_loss_fct(
            gen_logits.view(-1, gen_logits.size(-1)), 
            ods.reshape(-1)
        )
        # token acc
        chosen_tokens = torch.max(gen_logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.reshape(-1) == ods.reshape(-1)).to(torch.long)
        valid_mask = (ods != self.pad).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc

    @torch.no_grad()
    def calculate_ppl(self, batch):
        self.model.eval()
        ids, ids_mask = batch['ids'], batch['ids_mask']
        gen_logits = self.model(input_ids=ids, attention_mask=ids_mask).logits
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl
    
    @torch.no_grad()
    def evaluate(self, batch_size=32, quiet=True):
        data_dir = f'/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_all_phrase_ref_check_valid_merged/dev'
        gpt_batch, gpt_label = [], []
        query_text, phrase_pos = [], []
        with open(f'{data_dir}/dev_query_tok.jsonl') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = json.loads(line)
                index, tok_label, query = line['index'], line['results'], line['query']
                toks = []
                for tok, label, pos in tok_label:
                    toks.append(tok)
                gpt_batch.append(toks)
        acc_cnt = 0
        total_cnt = 0
        for st in tqdm(range(0, len(gpt_batch), batch_size), desc='Encoding queries', disable=quiet):
            end = min(st + batch_size, len(gpt_batch))
            gpt2_ids = gpt_batch[st: end]
            gpt2_ids = pad_sequence([torch.LongTensor(i) for i in gpt2_ids], padding_value=self.tokenizer.eos_token_id, batch_first=True).cuda()
            gpt2_mask = generate_mask(gpt2_ids, pad_token_idx=self.tokenizer.eos_token_id).cuda()
            ids, ods = gpt2_ids[:, :-1], gpt2_ids[:, 1:]
            ids_mask = gpt2_mask[:, :-1]

            output = self.model(input_ids=ids, attention_mask=ids_mask)
            gen_logits = output.logits
            # token acc
            chosen_tokens = torch.max(gen_logits, dim=-1)[1]    # [B, S-1]
            gen_acc = (chosen_tokens.reshape(-1) == ods.reshape(-1)).to(torch.long)
            valid_mask = (ods != self.pad).reshape(-1)
            valid_tokens = gen_acc & valid_mask
            acc_cnt += valid_tokens.sum().item()
            total_cnt += valid_mask.sum().item()
        gen_acc = acc_cnt / total_cnt
        return gen_acc
