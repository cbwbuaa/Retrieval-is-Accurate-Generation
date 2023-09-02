import transformers, torch, faiss
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from mips import MIPS

#Load GPT2
tokenizer = GPT2Tokenizer.from_pretrained('/apdcephfs/share_916081/ponybwcao/PLM/GPT2-small')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('/apdcephfs/share_916081/ponybwcao/PLM/GPT2-small', return_dict=True)
model = model.eval()
model = model.cuda()

#Set datastore
MAXIMUM_SIZE = 10000
DIMENSION = 768
STORE_FILE = 'keys.npy'
all_keys = np.memmap(STORE_FILE, dtype=np.float32, mode='w+', shape=(MAXIMUM_SIZE, DIMENSION))
finished_keys = 0

TOKEN_FILE='tokens.npy'
all_tokens = np.memmap(TOKEN_FILE, dtype=np.int, mode='w+', shape=(MAXIMUM_SIZE,))
all_lengths = []
finished_tokens = 0


# suppose we have the following data
data = [batch] * 10

# encode data
for batch in data:
    inputs = tokenizer(batch, padding=True, return_length=True, return_tensors="pt")
    assert (inputs['length'] > 1).all()
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'],
                        attention_mask = inputs['attention_mask'],
                        output_hidden_states=True)
        # We pick the hidden state at the last layer as the key
        keys = outputs['hidden_states'][-1]
        bsz, seq_len, dim = keys.shape
        for i in range(bsz):
            len_i = inputs['length'][i]
            all_keys[finished_keys: finished_keys+len_i-1] = keys[i, :len_i-1] # we do not need the last key 
            all_tokens[finished_tokens: finished_tokens+len_i-1] = inputs['input_ids'][i, 1: len_i]
            finished_keys += (len_i -1)
            finished_tokens += len_i
        all_lengths.extend(inputs['length'].tolist())
    # print ('finished_keys', finished_keys, 'finished_tokens', finished_tokens)

print('Encoding done!')




# # make index
# INDEX_TYPE = "Flat" # change it to 'IVF4096_HNSW32,SQ8' or whatever when dealing with big data
# mips = MIPS(DIMENSION, INDEX_TYPE, efSearch=128, nprobe=64)
# mips.train(all_keys[:finished_keys])
# mips.add(all_keys[:finished_keys])
# cumsum_keys = np.cumsum(np.array(all_lengths)-1)


# # save everything
# CUMSUM_KEYS_FILE = 'cumsum.npy'
# MIPS_INDEX_FILE = 'mips.index'
# np.save(open(CUMSUM_KEYS_FILE, 'wb'), cumsum_keys)
# mips.save(MIPS_INDEX_FILE)

# print ("preprocess done!")