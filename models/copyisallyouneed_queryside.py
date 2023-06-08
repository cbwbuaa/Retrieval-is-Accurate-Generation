from header import *
import numpy as np
import faiss
import random
import faiss.contrib.torch_utils
# torch.set_printoptions(profile='full')

def load_emb(path='embeddings.pkl'):
    with open(path, "rb") as fIn:
        stored_data = pickle.load(fIn)
    return stored_data

class CopyisallyouneedQueryside(nn.Module):

    def __init__(self, **args):
        super(CopyisallyouneedQueryside, self).__init__()
        self.args = args

        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args['prefix_encoder_tokenizer'][self.args['lang']])
        self.vocab_size = len(self.tokenizer)
        self.pad = self.tokenizer.pad_token_id if self.args['lang'] == 'zh' else self.tokenizer.bos_token_id

        self.model = GPT2LMHeadModel.from_pretrained(self.args['prefix_encoder_model'][self.args['lang']])
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])

        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)

        index_path = f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/phrase_index/best_prebatch_neg0_pretrain40w_1000000/IP_ivf_pq_index.faiss'
        # index_path = f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/phrase_index/best_prebatch_neg0_pretrain40w_1000000/IP_ivf128pq96_index.faiss'
        print('load index from', index_path)
        self.retriever = faiss.read_index(index_path)
        index_ivf = faiss.extract_index_ivf(self.retriever)
        index_ivf.make_direct_map()
        index_ivf.set_direct_map_type(faiss.DirectMap.Hashtable)
        self.retriever.nprobe = 128
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        self.gpu_retriever = faiss.index_cpu_to_gpu(res, 0, self.retriever, co)
        self.phrase_list = load_emb('/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/phrase_index/best_prebatch_neg0_pretrain40w_1000000/cluster_text_list.pkl')
        # self.retriever = faiss.index_cpu_to_gpu(res, 0, retriever)
        # self.retriever.efSearch = 128
        # print(self.retriever.ntotal) # 3796030

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return output

    def get_token_loss(self, ids, hs, ids_mask):
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
        ## gpt2 query encoder
        ids, ids_mask = batch['gpt2_ids'], batch['gpt2_mask']
        last_hidden_states = \
        self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True).hidden_states[-1]
        # get token loss
        loss_0, acc_0 = self.get_token_loss(ids, last_hidden_states, ids_mask)
        # collect the query representations
        query = last_hidden_states[:, :-1].reshape(-1, last_hidden_states.size(-1))
        # D, I, R = self.retriever.search_and_reconstruct(query, self.args['candidate_phrase_num'])
        D, I = self.gpu_retriever.search(query, self.args['candidate_phrase_num'])
        # I = I.cuda()
        # R = R.cuda()
        # # sanity check
        # print(R.shape)
        # valid_mask = (I != -1).reshape(-1)
        # valid_mask = valid_mask.expand(query.shape[0], valid_mask.shape[-1]).to(torch.long)
        # logits = torch.matmul(query, R.view(-1, R.shape[-1]).t())
        # new_logits = torch.where(valid_mask.to(torch.bool), logits, torch.tensor(-1e4).to(torch.half).cuda())
        # x = torch.argsort(new_logits, dim=-1, descending=True)[:, :10]
        # print(x)
        # exit()

        labels = batch['target_idxs'][:, 1:].reshape(-1)
        label_idxs = []
        valid_mask = (I != -1).reshape(-1).to(torch.long)
        total_phrase_num = I.size(0) * I.size(1)
        query_num = query.size(0)
        pos_mask = torch.zeros(query_num, total_phrase_num).to(torch.long).cuda()
        chunk_size = I.size(1)
        cur_pos = self.vocab_size
        # v1
        for i, (candidates, label) in enumerate(zip(I, labels)):
            start_pos = i * chunk_size
            end_pos = i * chunk_size + chunk_size 
            pos_mask[i, start_pos: end_pos] = 1
            if label < self.vocab_size: # token
                label_idx = label
            else: # phrase
                real_label = label - self.vocab_size
                label_idx = torch.nonzero(candidates == real_label)
                if label_idx.shape[0] != 0:
                    label_idx = label_idx[0][0] + cur_pos
                else:
                    label_idx = len(candidates) - 1
                    I[i][label_idx] = real_label
                    # R[i][label_idx] = self.retriever.reconstruct(real_label.item()).cuda()
                    label_idx += cur_pos
            cur_pos += len(candidates)
            label_idxs.append(label_idx)
        R = [self.retriever.reconstruct(x.item()) for x in I.view(-1)]
        R = torch.vstack(R).cuda()
        # v2
        # phraseidx2pos_map = {}
        # for i, (embs, candidates, label) in enumerate(zip(R, I, labels)):
        #     for emb, phrase_idx in zip(embs, candidates):
        #         if phrase_idx != -1:
        #             if phrase_idx not in phraseidx2pos_map:
        #                 cand_reps.append(emb)
        #                 phraseidx2pos_map[phrase_idx] = cur_pos
        #                 cur_pos += 1
        #     if label < self.vocab_size: # token
        #         label_idx = label
        #     else:
        #         real_label = label.item() - self.vocab_size
        #         if real_label not in phraseidx2pos_map:
        #             cand_reps.append(self.retriever.reconstruct(real_label))
        #             phraseidx2pos_map[real_label] = cur_pos
        #             label_idx = cur_pos
        #             cur_pos += 1
        #         else:
        #             label_idx = phraseidx2pos_map[real_label]
        #     label_idxs.append(label_idx)
        
        # cand_reps = torch.vstack(cand_reps)
        # cand_reps = cand_reps.view(-1, cand_reps.shape[-1]).cuda()
        # assert torch.isfinite(cand_reps).all()
        candidate_reps = torch.vstack((self.token_embeddings, R))
        # candidate_reps = torch.vstack((self.token_embeddings, R.view(-1, R.shape[-1])))
        logits = torch.matmul(query, candidate_reps.t())
        logits /= self.args['temp']
        
        overall_mask = torch.ones_like(logits).to(torch.long)
        overall_mask[:, self.vocab_size:] = valid_mask * pos_mask

        label_idxs = torch.tensor(label_idxs).cuda()
        # build the padding mask for query side
        query_padding_mask = ids_mask[:, :-1].reshape(-1).to(torch.bool)
        
        # new_logits = logits
        new_logits = torch.where(overall_mask.to(torch.bool), logits, torch.tensor(-1e4).to(torch.half).cuda())
        
        # assert (new_logits[query_padding_mask] != 0).sum() > 0
        # assert torch.isfinite(new_logits[query_padding_mask]).all()
        # new_logits = torch.where(position_mask, logits, torch.tensor(-1e4).to(torch.half).cuda())
        mask = torch.zeros_like(new_logits)
        mask[range(len(new_logits)), label_idxs] = 1.
        loss_ = F.log_softmax(new_logits[query_padding_mask], dim=-1) * mask[query_padding_mask]
        loss_1 = (-loss_.sum(dim=-1)).mean()

        # retrieve acc
        retrieve_acc = new_logits[query_padding_mask].max(dim=-1)[1] == label_idxs[query_padding_mask]
        retrieve_acc = retrieve_acc.to(torch.float).mean().item()
        # phrase_indexes = label_idxs > self.vocab_size
        # phrase_indexes_ = phrase_indexes & query_padding_mask
        # phrase_acc = new_logits[phrase_indexes_].max(dim=-1)[1] == label_idxs[phrase_indexes_]
        # phrase_acc = phrase_acc.to(torch.float).mean().item()

        # phrase_indexes_ = ~phrase_indexes & query_padding_mask
        # token_acc = new_logits[phrase_indexes_].max(dim=-1)[1] == label_idxs[phrase_indexes_]
        # token_acc = token_acc.to(torch.float).mean().item()
        return (
            loss_0,  # token loss
            loss_1,  # phrase loss
            acc_0,  # token accuracy
            retrieve_acc
        )

    @torch.no_grad()
    def encode_batch(self, sentences, batch_size=256, device='cuda'):
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

            output = self.phrase_encoder(**{k: v for k, v in encoded_dict.items() if k != 'offset_mapping'}, output_hidden_states=True)['hidden_states'][-1]  # [B, S, E]
            # collect the phrase start representations and phrase end representations
            s_rep = self.s_proj(output)
            e_rep = self.e_proj(output)
            all_s_rep.extend(s_rep)
            all_e_rep.extend(e_rep)
        return all_s_rep, all_e_rep, all_offsets

