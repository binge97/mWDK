import copy
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score,normalized_mutual_info_score,accuracy_score
from munkres import Munkres
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
import clustering_methods as cmd
from utils import GSNN, load_data,WL, WL_noconcate_one,WL_noconcate, IGK_WL_noconcate,IK_inne_fm,IK_fm_dot,WL_noconcate_gcn,pplot2,pplot3,pplot_single
from utils import create_adj_avg,WL_noconcate_fast,create_adj_avg_gcn,create_adj_avg_sp,adj_plot
from sklearn.kernel_approximation import Nystroem
import warnings
import scipy.io as sio
from sklearn import preprocessing
from Lambda_feature import lambda_feature_continous
from sklearn.metrics.pairwise import pairwise_distances
import argparse


warnings.filterwarnings('ignore')

def to_onehot(prelabel):
    k = len(np.unique(prelabel))
    label = np.zeros([prelabel.shape[0], k])
    label[range(prelabel.shape[0]), prelabel] = 1
    label = label.T
    return label
def square_dist(prelabel, feature):
    if sp.issparse(feature):
        feature = feature.todense()
    feature = np.array(feature)


    onehot = to_onehot(prelabel)

    m, n = onehot.shape
    count = onehot.sum(1).reshape(m, 1)
    count[count==0] = 1

    mean = onehot.dot(feature)/count
    a2 = (onehot.dot(feature*feature)/count).sum(1)
    pdist2 = np.array(a2 + a2.T - 2*mean.dot(mean.T))

    intra_dist = pdist2.trace()
    inter_dist = pdist2.sum() - intra_dist
    intra_dist /= m
    inter_dist /= m * (m - 1)
    return intra_dist

def smooth(embedding,labels):
    embs=IK_fm_dot(embedding,32,100)
    resi,resj,dis=-1,-1,np.inf
    # sim = pairwise_distances(embedding, metric="cosine")
    sim=embs.dot(embs.T)
    class0= [i for i in range(len(labels)) if labels[i]==0]
    class1= [i for i in range(len(labels)) if labels[i]==1]
    class2=[i for i in range(len(labels)) if labels[i]==2]
    class3= [i for i in range(len(labels)) if labels[i]==3]
    class4=[i for i in range(len(labels)) if labels[i]==4]
    class5= [i for i in range(len(labels)) if labels[i]==5]
    class6=[i for i in range(len(labels)) if labels[i]==6]
    # for i in class1:
    #     for j in class2:
    #         if sim[i][j]<dis:
    #             dis =sim[i][j]
    #             resi=i
    #             resj =j

    s = [embs[j] for j in class0]
    s0 = np.mean(s, axis=0)
    s=[embs[i] for i in class1]
    s1=np.mean(s,axis=0)
    s=[embs[j] for j in class2]
    s2=np.mean(s,axis=0)
    s = [embs[j] for j in class3]
    s3 = np.mean(s, axis=0)
    s = [embs[j] for j in class4]
    s4 = np.mean(s, axis=0)
    s = [embs[j] for j in class5]
    s5 = np.mean(s, axis=0)
    s = [embs[j] for j in class6]
    s6 = np.mean(s, axis=0)


    a0,a1,a2,a3,a4,a5,a6=0,0,0,0,0,0,0
    for i in class1:
        a0 += embs[i].dot(s0.T)
    for i in class1:
        a1 += embs[i].dot(s1.T)
    for i in class1:
        a2 += embs[i].dot(s2.T)
    for i in class1:
        a3 += embs[i].dot(s3.T)
    for i in class1:
        a4 += embs[i].dot(s4.T)
    for i in class1:
        a5 += embs[i].dot(s5.T)
    for i in class1:
        a6 += embs[i].dot(s6.T)

    ss=[s0,s1,s2,s3,s4,s5,s6]
    ss=np.array(ss)
    ss_all=np.mean(embs,axis=0)
    # ss=preprocessing.normalize(ss,"l2")
    ss=ss.dot(ss_all.T)







    # dis2=np.mean(sim)
    # return dis,resi,resj,dis2,dis/dis2
    return np.mean(ss),a0/len(class0),a1/len(class1),a2/len(class2),a3/len(class3),a4/len(class4),a5/len(class5),a6/len(class6)
def get_neigbors(g, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1, depth + 1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
    return output

def weight(G,embedding):
    # mean_emb = np.zeros_like(embedding,dtype=float)
    # neighbors_list = [get_neigbors(G, tar, depth=1)[1] for tar in range(node_features.shape[0])]
    # for i in range(node_features.shape[0]):
    #     neighbors = neighbors_list[i]
    #     mean_emb[i] = np.mean(embedding[neighbors],axis=0)
    # mean_emb=preprocessing.normalize(mean_emb,"l2")
    # sim =mean_emb.dot(mean_emb.T)
    # np.fill_diagonal(sim,0)
    # sim = preprocessing.normalize(sim, "l2",axis=1)
    #
    # sim = 0.5*np.log2((1-sim)/sim)
    # # sim =1/sim
    # sim = np.where(sim==np.inf,0,sim)

##################################################### version2
    # mean_emb = np.mean(embedding,axis=0)
    # sim = embedding.dot(mean_emb)
    # sim =(1+np.exp(sim))
    # #
    #
    # sim=np.diag(sim)
#################################################### version3
    mean_emb = np.zeros_like(embedding, dtype=float)
    neighbors_list = [get_neigbors(G, tar, depth=1)[1] for tar in range(node_features.shape[0])]
    for i in range(node_features.shape[0]):
        neighbors = neighbors_list[i]
        mean_emb[i] = np.mean(embedding[neighbors], axis=0)

    mean_emb = IK_fm_dot(mean_emb,32,100)
    sim = mean_emb.dot(mean_emb.T)
    # sim =np.diag(np.diag(sim))

    return sim,mean_emb


def normalize_adj(adj, type='rw'):
    np.fill_diagonal(adj, 1)
    """Symmetrically normalize adjacency matrix."""

    if type == 'sym':

        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        return adj*d_inv_sqrt*d_inv_sqrt.flatten()
        # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # d_mat_inv_sqrt = sp.diags(d_inv_sqrt).todense()
        # return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized
    elif type == 'wl':


        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = np.diag(d_inv)
        adj_normalized = adj.dot(d_mat_inv)
        return adj_normalized

def create_adj_avg_temp(adj_mat,sim):
    '''
    create adjacency
    '''
    np.fill_diagonal(adj_mat, 0)
    adj = copy.deepcopy(adj_mat)
    deg = np.sum(adj, axis=1)
    deg[deg == 0] = 1
    deg = (1/ deg) * 0.5
    deg_mat = np.diag(deg)
    adj = deg_mat.dot(adj_mat)
    # np.fill_diagonal(sim,0)
    # sim = preprocessing.normalize(sim,'l2')
    adj = adj *sim
    np.fill_diagonal(adj,0.5)
    return adj

def group_partition(predict_labels):
## 获取每个group的下标
    num_of_class = np.unique(predict_labels).shape[0]
    group_of_pre = []
    for i in range(num_of_class):
        temp = np.where(predict_labels == i)[0].tolist()
        group_of_pre.append(temp)
    return group_of_pre

def WL_max(embedding,adj,gg):
    new_emb = copy.deepcopy(embedding)
    embedding=embedding.astype(float)
    for i in range(adj.shape[0]):
        nei = get_neigbors(gg,i)[1]+[i]

        temp = embedding[nei]
        res = np.max(temp,axis=0)
        new_emb[i] = res
    return new_emb.astype(float)
def WL_min(embedding,adj,gg):
    new_emb = copy.deepcopy(embedding)
    embedding=embedding.astype(float)
    for i in range(adj.shape[0]):
        nei = get_neigbors(gg,i)[1]+[i]

        temp = embedding[nei]
        res = np.min(temp,axis=0)
        new_emb[i] = res
    return new_emb.astype(float)
if __name__ == '__main__':
    parser = argparse.ArgumentParser("mWDK")

    parser.add_argument("--dataset", type=str, default='cora', help="graph dataset")
    parser.add_argument("--m", type=str, default='mWDK', help="the embedding method")
    parser.add_argument("--psi", type=int, default=64, help="the number of samples for IK")
    parser.add_argument("--t", type=int, default=150, help="the number of sampling for IK")
    parser.add_argument("--h", type=int, default=15, help="the maximum itertion for WDK and mWDK")
    parser.add_argument("--path", type=str, default='./dataset/real_world data/', help="the path of datasets")

    args = parser.parse_args()

    adj_mat, node_features, true_labels = load_data(args.path, args.dataset)
    num_of_class = np.unique(true_labels).shape[0]

    if args.m == 'WL':
        num_of_class = np.unique(true_labels).shape[0]
        acc_li,nmi_li,f1_li = [],[],[]
        embedding = node_features.copy()
        new_adj = create_adj_avg(adj_mat)

        for h in range(100):
            embedding= WL_noconcate_fast(embedding,new_adj)
            acc,nmi,f1,para,predict_labels = cmd.ikbc(embedding,1,num_of_class,true_labels)

    if args.m == 'mWDK':
            myli = [i for i in range(len(node_features))]
            adj_mat = np.where(adj_mat!=0,1.0,0)
            new_adj = create_adj_avg(adj_mat)
            num_of_class = np.unique(true_labels).shape[0]
            embedding = node_features.copy()
            time_start = time.perf_counter()
            for h in range(1,15):
                embedding = IK_fm_dot(embedding, args.psi, t=args.t)
                embedding = WL_noconcate_fast(embedding, new_adj)
                # emb = preprocessing.normalize(embedding, norm='l2',axis=1)
                acc, nmi, f1, para, predict_labels = cmd.sc_linear(embedding, 1, num_of_class, true_labels)
                print('@{} psi={} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(args.dataset,args.psi,h,para,acc,nmi,f1))


    if args.m == 'WDK':
        new_adj = create_adj_avg(adj_mat)
        embedding = IK_fm_dot(embedding, args.psi, t=args.t)
        for h in range(31):
            embedding = WL_noconcate_fast(embedding, new_adj)
            acc,nmi,f1,para,predict_labels = cmd.sc_linear(embedding,1,num_of_class,true_labels)
            print('@{} psi={} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(args.dataset, args.psi, h, para, acc, nmi, f1))
