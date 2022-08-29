import numpy as np
import faiss                   # make faiss available
d = 32                         # dimension

data_path  = '/media/lab/2.2T/128D_Mem/128D_Sec_Clu_{}_Mem.npy'
ID_path    = '/media/lab/2.2T/Sec_Idx/Sec_Clu_{}_MemIdx.npy' 
score_path = '/media/lab/2.2T/hubness_score/hubness_score_{}.npy'

new_data_path = '/media/lab/2.2T/hubness_128D_Mem/hubness_128D_Sec_Clu_{}_Mem.npy'
new_ID_path   = '/media/lab/2.2T/hubness_Sec_Idx/hubness_Sec_Clu_{}_MemIdx.npy' 

persioteage = 0.75

for i in range (4096):
    data  = np.load(data_path.format(str(i)))
    ID    = np.load(ID_path.format(str(i)))
    score = np.load(score_path.format(str(i)))
    # score = np.int64(dataset)

    score_index = np.argsort(score)

    sort_index = score_index[:persioteage*data.shape[0]]
    # sort_index = np.sort(sort_index)[::-1]
    # for want_to_del in sort_index:
    ID = np.delete( ID, sort_index, 0 )
    data = np.delete( data, sort_index, 0 )

    np.save(new_ID_path.format(str(i)),ID)
    np.save(new_data_path.format(str(i)),data)

    






    