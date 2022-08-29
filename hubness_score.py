import numpy as np
import faiss                   # make faiss available
import math
d = 32                         # dimension


def gaussian(x, sig, avg):
    return (1./(math.sqrt(sig**math.pi))*np.exp(-sig*np.power((x - avg), 2.)))

data_path    = '/media/lab/2.2T/pagerank_data/Gaussian_hubness8_top1000_{}.npy'#'./input/Sec_Mem/Sec_Clu_{}_Mem.npy'
save_path    = './input/hubness_score/hubness_score_{}.npy'
ID_save_path = './input/data_graph/data_graph_{}.npy'
knn = 8
res = faiss.StandardGpuResources()
std = 0.75

for i in range (4096):
    dataset = np.load(data_path.format(str(i)))
    dataset = np.float32(dataset)
    score = np.zeros(dataset.shape[0])

    # build the index
    index = faiss.IndexFlatL2(d)
    # Set GPU
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
    # add vectors to the index
    # print(dataset.shape)
    gpu_index_flat.add(dataset)
    # search
    D, I = gpu_index_flat.search(dataset, knn) # sanity check
    # Point
    # for j in range(I.shape[0]):
    #     for k in range(I.shape[1]):
    #         score[I[j,k]] += gaussian(D[j,k], std, 0)

    np.save(ID_save_path.format(str(i)), I)
    # np.save(save_path.format(str(i)), score)