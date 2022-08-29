import numpy as np


data_path  = '/media/lab/2.2T/128D_Mem/128D_Sec_Clu_{}_Mem.npy'
score_path = './input/hubness_score/hubness_score_{}.npy'#'/media/lab/2.2T/hubness_score16/hubness_score_{}.npy'
new_data_path = '/media/lab/2.2T/Gaussian05_hubness8_round_1M.npy'

max_num = 1000000
data_1M = np.zeros((max_num,128),dtype=int)
index = 0

for i in range (4096):
    # Load data
    data  = np.load(data_path.format(str(i)))
    score = np.load(score_path.format(str(i)))
    # Total data
    Ndata = data.shape[0]
    # Sort
    score_index = np.argsort(score)
    # high score first
    score_index = score_index[::-1]
    sort_index = score_index[:round(Ndata/1000)]

    if i == 4095:
        sort_index = score_index[:(max_num-index)]
        print(max_num-index)

    for j in range (sort_index.shape[0]):
        data_1M[index+j] =  data[sort_index[j]]
    index += sort_index.shape[0]

print(index)
np.save(new_data_path,data_1M)