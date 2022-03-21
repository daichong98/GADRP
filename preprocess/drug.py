#基于药物的分子指纹构建相似性网络
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import torch
import csv
import heapq
#1448 drugs 881 dim
from sklearn.preprocessing import MinMaxScaler

drug_physicochemical_file = "/home/wh/python_file/gnnbased/data/drug/269-dim-physicochemical.csv"
drug_fingerprint_file= "/home/wh/python_file/gnnbased/data/drug/881-dim-fingerprint.csv"
drug_index_file="/home/wh/python_file/gnnbased/data/drug/drug_index.pt"
drug_fingerprint_sim_file= "/home/wh/python_file/gnnbased/data/drug/drug_fingerprint_sim.pt"
drug_fingerprint_sim_txt= "/home/wh/python_file/gnnbased/data/drug/drug_fingerprint_sim.txt"

drug_physicochemical_sim_file= "/home/wh/python_file/gnnbased/data/drug/drug_physicochemical_sim.pt"
drug_physicochemical_sim_txt= "/home/wh/python_file/gnnbased/data/drug/drug_physicochemical_sim.txt"


drug_sim_file= "/home/wh/python_file/gnnbased/data/drug/drug_sim.pt"
drug_sim_txt= "/home/wh/python_file/gnnbased/data/drug/drug_sim.txt"

drug_sim_top10_file="/home/wh/python_file/gnnbased/data/drug/drug_sim_top10.pt"
drug_sim_top10_txt="/home/wh/python_file/gnnbased/data/drug/drug_sim_top10_matrix.txt"


def main():
    #load drug
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    physicochemical_feature = pd.read_csv(drug_physicochemical_file, sep=',', header=0, index_col=[0])
    min_max = MinMaxScaler()
    physicochemical_feature = min_max.fit_transform(physicochemical_feature)

    pubcheid2fingerprint = pd.read_csv(drug_fingerprint_file, sep=',', header=0, index_col=[0])
    drugset=list(pubcheid2fingerprint.index)
    drugset=np.array(drugset)
    torch.save(drugset,drug_index_file)
    drug_num=len(drugset)
    pubcheid2fingerprint=pubcheid2fingerprint.values
    """
    drug_fingerprint_sim=torch.zeros(size=(drug_num,drug_num)).to(device)
    drug_physicochemical_sim = torch.zeros(size=(drug_num, drug_num)).to(device)

    # drug_fingerprint_sim=interaction/union
    for i in range(len(drugset)):
        print(i)
        for j in range(len(drugset)):
            temp=(pubcheid2fingerprint[i, :]+pubcheid2fingerprint[j, :])
            intersection=np.sum(temp==2)
            union=np.sum(temp>=1)
            drug_fingerprint_sim[i][j]=intersection/union

            temp_sim = pearsonr(physicochemical_feature[i, :], physicochemical_feature[j, :])
            drug_physicochemical_sim[i][j] = np.abs(temp_sim[0])
        #对每种药物选出相似度最高的前10个药物并存储
        #
    #drug_fingerprint_sim
    torch.save(drug_fingerprint_sim, drug_fingerprint_sim_file)
    np.savetxt(drug_fingerprint_sim_txt, drug_fingerprint_sim.cpu().numpy(), fmt='%4f', delimiter=",")
    #drug_physicochemical_sim
    torch.save(drug_physicochemical_sim, drug_physicochemical_sim_file)
    np.savetxt(drug_physicochemical_sim_txt, drug_physicochemical_sim.cpu().numpy(), fmt='%4f', delimiter=",")
    """
    #drug_sim_top10
    drug_sim=torch.load(drug_physicochemical_sim_file)
    _,drug_sim_top10 = torch.topk(drug_sim, 10,dim=1)
    # print(temp.shape)
    # drug_sim_top10 = torch.zeros(size=(drug_num, 10), dtype=torch.int).to(device)
    # for i in range(5):
    #     print(i)
    #     drugi_list = list(drug_sim[i])
    #     # print(heapq.nlargest(10, drugi_list))
    #     drug_sim_top10[i] = torch.tensor(list(map(drugi_list.index, heapq.nlargest(10, drugi_list))),dtype=torch.int)
    #     print(drug_sim_top10[i] )
    #     _,temp=torch.topk(drug_sim[i],10)
    #     print(temp)

    torch.save(drug_sim, drug_sim_file)
    np.savetxt(drug_sim_txt, drug_sim.cpu().numpy(), fmt='%4f', delimiter=",")

    #drug 相似性 top 10 的drug 的存储
    torch.save(drug_sim_top10,drug_sim_top10_file)
    np.savetxt(drug_sim_top10_txt,drug_sim_top10.cpu().numpy(),fmt='%d',delimiter=",")



if __name__ == '__main__':
    main()