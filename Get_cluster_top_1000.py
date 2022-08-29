import numpy as np


data_path  = '/media/lab/2.2T/Sec_Mem/Sec_Clu_{}_Mem.npy'
score_path = '/media/lab/2.2T/Gaussian_hubness_score8/hubness_score_{}.npy'#'./input/hubness_score/hubness_score_{}.npy'#
data_for_pagerank = '/media/lab/2.2T/pagerank_data/Gaussian_hubness8_top1000_{}.npy'





for i in range (4096):

    # print(i)
    data_get = np.zeros((1000,32),dtype=int)
    data  = np.load(data_path.format(str(i)))
    score = np.load(score_path.format(str(i)))
    score_index = np.argsort(score)
    score_index = score_index[::-1]
    sort_index = score_index[:1000]

    for j in range (1000):
        data_get[j] =  data[sort_index[j]]
    # print(score[0])
    np.save(data_for_pagerank.format(str(i)),data_get)



# data_1M = np.asarray(data_1M)
# print(data_1M.dtype)
