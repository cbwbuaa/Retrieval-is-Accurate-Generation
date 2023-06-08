import faiss
import numpy as np

# see http://ulrichpaquet.com/Papers/SpeedUp.pdf theorem 5
def augment_data(xb, phi=None, factor=1.0): 
    norms = np.linalg.norm(xb, axis=1, keepdims=True)
    if phi is None: 
        phi = norms.max() * factor
    extracol = np.sqrt(np.maximum(0.0, phi**2 - norms**2))
    return np.concatenate([extracol, xb], axis=1), phi

def augment_query(xq): 
    extracol = np.zeros((len(xq), 1), dtype=np.float32)
    return np.concatenate([extracol, xq], axis=1)

def l2_to_ip(l2_score, query, max_norm=None):
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    if max_norm is None:
        return -0.5 * (l2_score - query_norm ** 2)
    return -0.5 * (l2_score - query_norm ** 2 - max_norm ** 2)

class MIPS:
    def __init__(self, d, index_type=None, efSearch=None, nprobe=None):
        # simple index
        if index_type is not None:
            index = faiss.index_factory(d, index_type)
            if efSearch is not None:
                index.efSearch = efSearch
            if nprobe is not None:
                index.nprobe = nprobe
            self.index = index
        else:
            self.index = None
    
    def to_gpu(self, gpuid=0):
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, gpuid, self.index)

    def to_cpu(self):
        self.index = faiss.index_gpu_to_cpu(self.index)
 
    def train(self, data):
        self.index.train(data)

    def add(self, data):
        self.index.add(data)

    def add_with_ids(self, data, ids):
        self.index.add_with_ids(data, ids)

    def search(self, query, k):
        return self.index.search(query, k)

    def reconstruct(self, idx):
        return self.index.reconstruct(idx)

    def save(self, path):
        faiss.write_index(self.index, path)

    def reset(self):
        self.index.reset()
   
    @classmethod
    def from_built(cls, path, nprobe=None):
        index = faiss.read_index(path)
        if nprobe is not None:
            index.nprobe = nprobe
        mips = cls(index.d)
        mips.index = index
        return mips

if __name__ == '__main__':
    # SIZE = 1000000
    # DIM = 128
    # data = np.random.random((SIZE, DIM)).astype(np.float32)
    data_dir = '/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/phrase_index/best_0601_shuffle_queue5k_mergedQ_eval1k_dim128_focal_loss_lr1e-4_prebatch0_beta0.5_warmup50000_prenum0_temp2.0_400000'
    data = np.load(f'{data_dir}/phrase_emb.npy')
    
    # exact IP search
    d = data.shape[1]
    # golden_index = faiss.IndexFlatIP(d)
    # golden_index.add(data)

    data, max_norm = augment_data(data)

    # # exact L2 search
    # d = data.shape[1]
    # l2_index = faiss.IndexFlatL2(d)
    # l2_index.add(data)

    mips = MIPS(d, "OPQ128_256,IVF65536_HNSW32,PQ128x4fsr", efSearch=128, nprobe=256)
    # mips = MIPS(d, "IVF100_HNSW32,SQ8", efSearch=128, nprobe=64)
    #mips = MIPS(d, "IVF10_HNSW32,Flat", efSearch=128, nprobe=64)
    # mips.to_gpu()
    mips.train(data)
    mips.add(data)
    mips.save(f'{data_dir}/OPQ128_256,IVF65536_HNSW32,PQ128x4fsr.index')
    # mips.add(data[:SIZE//2])
    # mips.add(data[SIZE//2:])
    #mips.to_cpu()

    # query = np.random.random((1000, DIM)).astype(np.float32)
    # D0, I0 = golden_index.search(query, 1)
    # query = augment_query(query)
    # D1, I1 = l2_index.search(query, 1)
    # _, I2 = mips.search(query, 1)


    # def R1_metric(i, j):
    #     return (i == j).astype(np.float32).mean()


    # print ('L2&IP agreement')
    # print(R1_metric(I0, I1))
    # DD = l2_to_ip(D1, query, max_norm)
    # print (( np.abs(DD-D0) < 1e-5).all())
    # print ('########')
    # print(R1_metric(I0, I2))
    # print(R1_metric(I1, I2))
