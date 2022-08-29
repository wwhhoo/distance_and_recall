import numpy as np

d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss

nlist = 100
m = 8
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                  # 8 specifies that each sub-vector is encoded as 8 bits
index.train(xb)
for i in range(10):
    print(i)
    faiss.write_index(index, "index_IVFPQ.index") 
    index = faiss.read_index("index_IVFPQ.index")
    index.remove(xb)
    index.add(xb)
    D, I = index.search(xb[:5], k) # sanity check
    print(I)
    print(D)
    index.nprobe = 10              # make comparable with experiment above
    D, I = index.search(xq, k)     # search
    print(I[-5:])