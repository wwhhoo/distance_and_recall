import numpy as np
import faiss                   # make faiss available
import random

data_path = '/media/lab/2.2T/hubness_128D_Mem/hubness_128D_Sec_Clu_{}_Mem.npy'



for i in range (3520):
    data  = np.load(data_path.format(str(i)))
    num   = list(range(0,data.shape[0]))
    random.shuffle(num)
    for j in range (244):
        num[j]


for i in range (3520:4096):
    data  = np.load(data_path.format(str(i)))
    num   = list(range(0,data.shape[0]))
    random.shuffle(num)
    for j in range (245):

np.save(data,new_data_path.format(str(i)))


    






    