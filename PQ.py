import numpy as np
import faiss


def search_data_id( cluster, array_size, cluster_num, Ints_clustMem):
    sec          = -1
    ID_array     = np.zeros((array_size, 1))
    Mem128_array = np.zeros((array_size, 128))
    index        = 0
    # for all sec cluster in file cluster
    for sec_cluster in cluster:
        sec_cluster = int(sec_cluster)
        
        if sec_cluster != -1:
            if sec != int(sec_cluster / 4096):
                sec = int(sec_cluster / 4096)
                
                MemID_path  = '/home/lab/Documents/Refactor/input/Sec_Idx/Sec_Clu_{}_MemIdx.npy'.format(str(sec))
                Mem128_path = '/media/lab/2.2T/128D_Mem/128D_Sec_Clu_{}_Mem.npy'.format(str(sec))
                MemID       = np.load(MemID_path)
                Mem128      = np.load(Mem128_path)
            
            for targer_Clus_Num in range (int(cluster_num[sec_cluster])):
                MemID = np.int64(MemID)
                start = np.where(MemID == int(Ints_clustMem[sec_cluster][targer_Clus_Num]))
                ID_array[index + targer_Clus_Num]     = Ints_clustMem[sec_cluster][targer_Clus_Num]
                Mem128_array[index + targer_Clus_Num] = Mem128[start[0]]
            index += int(cluster_num[sec_cluster])

    return Mem128_array, ID_array

def countDataNumber(cand, clus):
    totalNum = 0
    for i in clus:
        if i == -1:
            break
        totalNum += cand[int(i)]
    return totalNum[0]

def recall(top, ad, label, faiss_I):
    hit_rate = 0.0
    for top_num in range(top):
        hit = np.where(faiss_I[0] == label[top_num])
        if len(hit[0]) > 0:
            hit_rate += 1

    return hit_rate

# def recall(top, ad, label, faiss_I):
#     hit = np.zeros(int(ad/10), dtype=int)
#     for top_num in range(top):
#         hit = np.where(faiss_I[0] == label[top_num])
#         if len(hit[0]) > 0:
#             if (hit[0] < 0 ):
#                 for i in range (int(ad/10)):
#                     hit[i] += 1
#             else if (hit[0] < 10 ):
#                 for i in range (1,int(ad/10)):
#                     hit[i] += 1
#             else if (hit[0] < 100 ):
#                 for i in range (2,int(ad/10)):
#                     hit[i] += 1
#             else if (hit[0] < 1000 ):
#                 hit[3] += 1

#     return hit

PQtrain_path  = ('./input/Gaussian_hubness16_average_1M.npy')
PQtest_path   = ('/home/lab/Documents/Refactor/input/AElabel.npy')
PQtrain       = np.load(PQtrain_path)
query         = np.loadtxt("./input/Test.txt")
label         = np.loadtxt("./input/Top1000_Test.txt", dtype=int)
memberNumber  = np.load("/home/lab/Documents/Refactor/input/Idx2CountMember.npy")
cluster_path  = "/home/lab/Documents/Refactor/input/Cand/QNN_M6_{}_10.txt"
Ints_clustMem = np.load("./input/Idx2Codebooks.npy",allow_pickle=True)

d     = 128                           # dimension
bits  = 8
nlist = 2**bits
m     = 16
k     = 1000


quantizer = faiss.IndexFlatL2(d)  # the other index
PQtrain   = np.float32(PQtrain)
print("train")

index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
index.train(PQtrain)
index_name = "index_PQ_Gaussian_hubness16_average_{}_{}.index".format(bits, nlist)
faiss.write_index(index, index_name)



for big_cluster in range(1):
    big_cluster  = 0
    cluster      = np.loadtxt(cluster_path.format(str(big_cluster+1)))
    recall_score = np.zeros(4)
    for times in range(10000):
        print(times)
        sec_clus = np.asarray(cluster[times])

        array_size = countDataNumber( memberNumber,sec_clus )
        totoal_128mem, totoal_mem_Id = search_data_id(sec_clus, array_size, memberNumber, Ints_clustMem)

        # Load code book
        index_ivfPQ = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
        index_ivfPQ = faiss.read_index(index_name)

        # candidate data & ID
        totoal_mem_Id = np.reshape(totoal_mem_Id, totoal_mem_Id.shape[0])
        totoal_128mem = np.float32(totoal_128mem)
        totoal_mem_Id = np.int64(totoal_mem_Id)

        index_ivfPQ.add_with_ids(totoal_128mem,totoal_mem_Id )          # add vectors to the index
        index_ivfPQ.nprobe = nlist

        query = np.float32(query)
        single_query = query[times]
        single_query = single_query.reshape((1,single_query.shape[0]))

        D, I = index_ivfPQ.search(single_query, k)     # search
        
        recall_score[0] += recall(1, 1, label[times], I[:,:1])
        recall_score[1] += recall(1, 10, label[times], I[:,:10])
        recall_score[2] += recall(1, 100, label[times], I[:,:100])
        recall_score[3] += recall(1, 1000, label[times], I)
        # recall_score[4] += recall(10, 10, label[times], I[:,:10])
        # recall_score[5] += recall(10, 100, label[times], I[:,:100])
        # recall_score[6] += recall(10, 1000, label[times], I)
        print(recall_score)
        # recall_score[7] += recall(100, 100, label[times], I[:,100], totoal_mem_Id)
        # recall_score[8] += recall(100, 1000, label[times], I, totoal_mem_Id)

    recall_score = recall_score/10000
    save_path = './output/recallPQ/PQ_Gaussian_hubness16_average_{}_{}_recall_{}_10.txt'.format(bits, nlist, str(big_cluster+1))
    np.savetxt(save_path, recall_score)
    print(index_name)
