import os
from rdkit import Chem
import numpy as np

import pandas as pd
from sklearn.preprocessing import scale,MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import torch
from sklearn.decomposition import PCA
import heapq
import csv



cell_copynumber_file = '/home/wh/python_file/gnnbased/data/cellline/copynumber_461cell_23316dim.csv'
cell_miRNA_file = '/home/wh/python_file/gnnbased/data/cellline/miRNA_470cell_734dim.csv'
cell_CpG_file = '/home/wh/python_file/gnnbased/data/cellline/CpG_407cell_69641dim.csv'
cell_RNAseq_file = '/home/wh/python_file/gnnbased/data/cellline/RNAseq_462cell_48392dim.csv'

cell_id_file="/home/wh/python_file/gnnbased/data/cellline/cell_index.csv"
RNAseq_feature_pca_file="/home/wh/python_file/gnnbased/data/cellline/RNAseq_feature_pca_file.csv"

copynumber_sim_file="/home/wh/python_file/gnnbased/data/cellline/copynumber_sim.pt"
copynumber_sim_matrix_file="/home/wh/python_file/gnnbased/data/cellline/copynumber_sim_matrix.csv"
miRNA_sim_file="/home/wh/python_file/gnnbased/data/cellline/miRNA_sim.pt"
miRNA_sim_matrix_file="/home/wh/python_file/gnnbased/data/cellline/miRNA_sim_matrix.csv"
CpG_sim_file="/home/wh/python_file/gnnbased/data/cellline/CPG_sim.pt"
CpG_sim_matrix_file="/home/wh/python_file/gnnbased/data/cellline/CPG_sim_matrix.csv"
RNAseq_sim_file="/home/wh/python_file/gnnbased/data/cellline/RNAseq_sim.pt"
RNAseq_sim_matrix_file="/home/wh/python_file/gnnbased/data/cellline/RNAseq_sim_matrix.csv"


cell_sim_file="/home/wh/python_file/gnnbased/data/cellline/cell_sim.pt"
cell_sim_matrix_file="/home/wh/python_file/gnnbased/data/cellline/cell_sim_matrix.csv"

cell_sim_top10_file="/home/wh/python_file/gnnbased/data/cellline/cell_sim_top10.pt"
cell_sim_top10_txt="/home/wh/python_file/gnnbased/data/cellline/cell_sim_top10_matrix.txt"


drug_cell_label_file = "../data/pair/drug_cell.csv"
cell_index_file="../data/cellline/cell_index.csv"


def main():
    """
    caculate cell line similarity network
    :return:
    """


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Load RNAseq feature  462*48392

    #计算四个相似性网络
    copynumber_feature = pd.read_csv(cell_copynumber_file, sep=',', header=None, index_col=[0], skiprows=5)
    miRNA_feature = pd.read_csv(cell_miRNA_file, sep=',', header=None, index_col=[0])
    CpG_feature = pd.read_csv(cell_CpG_file, sep=',', header=None, index_col=[0], skiprows=2)
    RNAseq_feature = pd.read_csv(cell_RNAseq_file, sep=',', header=None, index_col=[0], skiprows=2)

    cell_id_set = pd.read_csv(cell_id_file, sep=',', header=None, index_col=[0])
    copynumber_feature = copynumber_feature.loc[list(cell_id_set.index)].values
    miRNA_feature = miRNA_feature.loc[list(cell_id_set.index)].values
    CpG_feature = CpG_feature.loc[list(cell_id_set.index)].values
    RNAseq_feature = RNAseq_feature.loc[list(cell_id_set.index)].values

    # Normalization
    min_max=MinMaxScaler()
    copynumber_feature = min_max.fit_transform(copynumber_feature)
    miRNA_feature = min_max.fit_transform(miRNA_feature)
    CpG_feature = min_max.fit_transform(CpG_feature)
    RNAseq_feature = min_max.fit_transform(RNAseq_feature)


    copynumber_sim = torch.zeros(size=(len(copynumber_feature), len(copynumber_feature)))
    miRNA_sim = torch.zeros(size=(len(miRNA_feature), len(miRNA_feature)))
    CpG_sim = torch.zeros(size=(len(CpG_feature), len(CpG_feature)))
    RNAseq_sim = torch.zeros(size=(len(RNAseq_feature), len(RNAseq_feature)))


    for i in range(len(copynumber_feature)):
        for j in range(len(copynumber_feature)):
            temp_sim = pearsonr(copynumber_feature[i, :], copynumber_feature[j, :])
            copynumber_sim[i][j] = np.abs(temp_sim[0])

            temp_sim = pearsonr(miRNA_feature[i, :], miRNA_feature[j, :])
            miRNA_sim[i][j] = np.abs(temp_sim[0])

            temp_sim = pearsonr(CpG_feature[i, :], CpG_feature[j, :])
            CpG_sim[i][j] = np.abs(temp_sim[0])

            temp_sim = pearsonr(RNAseq_feature[i, :], RNAseq_feature[j, :])
            RNAseq_sim[i][j] = np.abs(temp_sim[0])
            # 对每种药物选出相似度最高的前10个药物并存储
        #celli_list = list(copynumber_sim[i])
        # cell_sim_top10[i] = torch.tensor(list(map(celli_list.index, heapq.nlargest(10, celli_list))),
        #                                  dtype=torch.int)
    torch.save(copynumber_sim, copynumber_sim_file)
    np.savetxt(copynumber_sim_matrix_file, copynumber_sim.numpy(), fmt='%4f', delimiter=",")

    torch.save(miRNA_sim, miRNA_sim_file)
    np.savetxt(miRNA_sim_matrix_file, miRNA_sim.numpy(), fmt='%4f', delimiter=",")

    torch.save(CpG_sim, CpG_sim_file)
    np.savetxt(CpG_sim_matrix_file, CpG_sim.numpy(), fmt='%4f', delimiter=",")

    torch.save(RNAseq_sim, RNAseq_sim_file)
    np.savetxt(RNAseq_sim_matrix_file, RNAseq_sim.numpy(), fmt='%4f', delimiter=",")


    #相似性网络的构建
    miRNA_sim = torch.load(miRNA_sim_file)
    CpG_sim=torch.load(CpG_sim_file)
    cell_sim=(miRNA_sim + CpG_sim)/2


    cell_sim_top10 = torch.zeros(size=(388, 10), dtype=torch.int).to(device)
    for i in range(388):
        celli_list = list(cell_sim[i])
        cell_sim_top10[i] = torch.tensor(list(map(celli_list.index, heapq.nlargest(10, celli_list))),
                                         dtype=torch.int)

    torch.save(cell_sim, cell_sim_file)
    np.savetxt(cell_sim_matrix_file, cell_sim.numpy(), fmt='%4f', delimiter=",")
    # drug 相似性 top 10 的drug 的存储
    torch.save(cell_sim_top10, cell_sim_top10_file)
    np.savetxt(cell_sim_top10_txt, cell_sim_top10.cpu().numpy(), fmt='%d', delimiter=",")





if __name__ == '__main__':
    main()