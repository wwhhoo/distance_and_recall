import numpy as np


data_path  = '/media/lab/2.2T/128D_Mem/128D_Sec_Clu_{}_Mem.npy'
score_path = './input/hubness_score/hubness_score_{}.npy'#'/media/lab/2.2T/Gaussian_hubness_score16/hubness_score_{}.npy'#
new_data_path = '/media/lab/2.2T/Gaussian05_hubness8_average_1M.npy'

data_1M = np.zeros((1000000,128),dtype=int)
index = 0


for i in range (3520):

    # print(i)
    data  = np.load(data_path.format(str(i)))
    score = np.load(score_path.format(str(i)))
    score_index = np.argsort(score)
    score_index = score_index[::-1]
    sort_index = score_index[:244]

    for j in range (244):
        data_1M[index+j] =  data[sort_index[j]]
    index += 244

for i in range (3520,4096):
    # print(i)
    data  = np.load(data_path.format(str(i)))
    score = np.load(score_path.format(str(i)))
    score_index = np.argsort(score)
    score_index = score_index[::-1]
    sort_index = score_index[:245]

    for j in range (245):
        data_1M[index+j] =  data[sort_index[j]]
    index += 245

# data_1M = np.asarray(data_1M)
# print(data_1M.dtype)
np.save(new_data_path,data_1M)