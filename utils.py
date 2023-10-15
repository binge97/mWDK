import copy
import pickle as pkl
import scipy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
# import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math
import random
from scipy.spatial.distance import pdist, squareform

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(path,dataset):
    data =sio.loadmat('{}{}.mat'.format(path,dataset))
    node_features = data['features']
    adj_mat = data['adj_mat']
    true_labels = data['labels'].reshape(-1)
    details = data['details'][0]
    print(details)

    ########################post processing####################
    # np.fill_diagonal(adj_mat,0)
    # deg =np.sum(adj_mat,axis=0)
    # ind_zero = np.where(deg==0)[0]
    # adj_mat=np.delete(adj_mat, ind_zero, axis=1)
    # adj_mat=np.delete(adj_mat, ind_zero, axis=0)
    # true_labels=np.delete(true_labels, ind_zero)
    # node_features=np.delete(node_features, ind_zero, axis=0)
    # np.fill_diagonal(adj_mat, 0)

    deg = np.sum(adj_mat, axis=0)
    # print('max_degree:{}    avergae_degree_{}'.format(np.max(deg), np.mean(deg)))


    return adj_mat, node_features, true_labels

def sub_wl(node_features, adj_mat1,adj_mat2, h):
    np.fill_diagonal(adj_mat2,1)

    list_of_index_of_neighbor=[]
    for i in range(adj_mat2.shape[0]):
        index_of_neighbor = [t for t in range(adj_mat2.shape[1]) if adj_mat2[i][t] != 0]
        list_of_index_of_neighbor.append(index_of_neighbor)
    embedding = []
    for id in range(len(list_of_index_of_neighbor)):
        sub_index = list_of_index_of_neighbor[id]
        tar = sub_index.index(id)
        sub_features = node_features[sub_index]
        i, j = np.ix_(sub_index,sub_index)
        sub_adj = adj_mat1[i, j]
        # sub_adj2 = adj_mat2[i,j]
        # sub_adj = sub_adj *sub_adj2
        # sub_adj[sub_adj!=0] = 1
        temp = WL_noconcate(sub_features,sub_adj,h)[tar]
        embedding.append(temp)
    return np.array(embedding)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

# def WL_noconcate(node_features, adj_mat,h):
#     node_features =[node_features]
#
#     # print("===================================@WL_noconcate at h={}===================================".format(h))
#     res = compute_wl_embeddings_continuous(h, node_features, adj_mat,)
#     X = res[0]
#     n = node_features[0].shape[1]
#     X= X[:, h * n:]
#     return X # no concate

# def WL(node_features, adj_mat,h):
#     print("===================================@WL at h={}===================================".format(h))
#     res = compute_wl_embeddings_continuous(h, node_features, adj_mat,)
#     X = res[0]
#
#     return X




# def IGK_WL_noconcate(node_features, adj_mat, h, psi,t,):
#     # print("===================================@IkWL at psi={} h={} t={}===================================".format(psi,h,t))
#     res = compute_wl_embeddings_continuous_by_IK(h, t, psi, node_features, adj_mat)
#     X = res[0]
#     n = node_features[0].shape[1]
#     X = X[:, h * psi*t:]
#     return X

def compute_wl_embeddings_continuous_by_IK(h, t, psi, node_features, adj_mat):
    node_features, adj_mat = node_features, adj_mat
    # transform into Isolation Vector
    partitions = generate_partitions(node_features, t, psi)
    cell_indexs = convert_to_cell_index(partitions, node_features)
    igk_vectors = convert_index_to_vector(cell_indexs, psi)
    # Generate the label sequences for h iterations
    labels_sequence = create_labels_seq_cont(igk_vectors, adj_mat, h)
    return labels_sequence

def compute_wl_embeddings_continuous(h, node_features, adj_mat):
    '''
    Continuous graph embeddings
    TODO: for package implement a class with same API as for WL
    '''

    # Generate the label sequences for h iterations
    labels_sequence = create_labels_seq_cont(node_features, adj_mat, h)
    return labels_sequence

def convert_to_cell_index(partitions, X):
    cell_indexs = []
    t = len(partitions)
    num_graph = len(X)
    X_concate = np.concatenate(X, axis=0)

    graph_cell_indexs = []
    for i in range(t):
        knn_model = partitions[i]
        cell_index = knn_model.predict(X_concate)
        graph_cell_indexs.append(cell_index)
    graph_cell_indexs = np.array(graph_cell_indexs)

    flag = 0
    for i in range(num_graph):
        point_num = X[i].shape[0]
        cell_indexs.append(graph_cell_indexs[:,flag:flag + point_num])
        flag += point_num
    return cell_indexs

def convert_index_to_vector(cell_indexs, psi):
    igk_vectors = []
    [t, _] = cell_indexs[0].shape
    for cell_index in cell_indexs:
        num_point = cell_index.shape[1]
        graph_vectors = np.zeros((num_point, psi * t))
        for i in range(num_point):
            instance_vector = np.zeros((1, psi * t))
            instance_index = cell_index[:,i].flatten()
            bias = [t * psi for t in range(t)]
            instance_index = instance_index + bias
            instance_vector[0, instance_index.flatten().tolist()] += 1
            graph_vectors[i,:] = instance_vector
        igk_vectors.append(graph_vectors)
    return igk_vectors

def generate_partitions(X, t, psi):
    # train iForest models
    X = np.concatenate(X, axis=0)
    knn_models = []
    for i in range(t):
        sample_index = np.random.permutation(len(X))
        sample_index = sample_index[0:psi]
        Y = [i for i in range(psi)]
        sample_X = X[sample_index, :]
        neigh = DecisionTreeClassifier(splitter='random')
        neigh.fit(sample_X, Y)
        knn_models.append(neigh)
    return knn_models

def create_labels_seq_cont(node_features, adj_mat, h):
    '''
    create label sequence for continuously attributed graphs
    '''
    n_graphs = len(node_features)
    labels_sequence = []
    for i in range(n_graphs):
        graph_feat = []
        for it in range(h + 1):
            if it == 0:
                graph_feat.append(node_features[i])
            else:

                adj_cur = adj_mat[i] + np.identity(adj_mat[i].shape[0])
                adj_cur = create_adj_avg(adj_cur)
                np.fill_diagonal(adj_cur, 0)
                graph_feat_cur = 0.5 * (np.dot(adj_cur, graph_feat[it - 1]) + graph_feat[it - 1])


                graph_feat.append(graph_feat_cur)
        labels_sequence.append(np.concatenate(graph_feat, axis=1))
    return labels_sequence

def create_adj_avg(adj_mat):
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
    np.fill_diagonal(adj, 0.5)
    return adj

def create_adj_avg_gcn(adj_mat):
    '''
    create adjacency
    '''
    adj = copy.deepcopy(adj_mat)
    np.fill_diagonal(adj, 1)
    deg = np.sum(adj, axis=1)
    deg[deg == 0] = 1
    deg = (1 / deg) * 0.5
    deg = np.sqrt(deg)

    deg_mat = np.diag(deg)

    adj_mat = adj_mat.dot(deg_mat).T
    adj_mat = adj_mat.dot(deg_mat)

    return adj_mat
def create_adj_avg_rw(adj_mat):
    '''
    create adjacency
    '''
    np.fill_diagonal(adj_mat, 0)

    adj = copy.deepcopy(adj_mat)
    deg = np.sum(adj, axis=1)
    deg[deg == 0] = 1
    deg = (1 / deg) * 0.5
    deg_mat = np.diag(deg)
    adj = deg_mat.dot(adj_mat).T
    np.fill_diagonal(adj, 0.5)
    return adj

def pplot(node_features,embedding, name, true_labels,new_predict_labels,G):
    embedding = np.around(embedding, 3)
    fig = plt.figure(figsize=(16, 5))

    ax2 = fig.add_subplot(131)
    ax2.scatter(node_features[:, 0], node_features[:, 1], c=true_labels, s=5, cmap="rainbow")
    tag = 0
    # for a, b in zip(node_features[:, 0], node_features[:, 1]):  # 添加这个循环显示坐标
    #     plt.text(a, b, (tag), ha='center', va='bottom', fontsize=7)
    #     tag+=1
    ax2.set_title("Source: Ground Truth " + name)
    plt.axis('equal')
    ax2 = fig.add_subplot(132, )
    ax2.scatter(embedding[:, 0], embedding[:, 1], c=true_labels, s=6, cmap="rainbow")
    tag = 0
    # for a, b in zip(embedding[:, 0], embedding[:, 1]):  # 添加这个循环显示坐标
    #
    #     plt.text(a, b, (tag), ha='center', va='bottom', fontsize=7)
    #     tag += 1
    ax2.set_title("Embedding: Ground Truth " + name)
    plt.axis('equal')
    ax2 = fig.add_subplot(133)
    ax2.scatter(embedding[:, 0], embedding[:, 1], c=new_predict_labels, s=6, cmap="rainbow")
    tag = 0
    # for a, b in zip(embedding[:, 0], embedding[:, 1]):  # 添加这个循环显示坐标
    #     plt.text(a, b, (a,b), ha='center', va='bottom', fontsize=10)
    #     tag += 1
    ax2.set_title("Embedding: PreLlabels " + name)
    plt.axis('equal')

    plt.show()
    plt.close()
    G = nx.from_numpy_matrix(adj_mat)

    pos = embedding_tsne
    plt.figure(dpi=200)

    nx.draw_networkx_nodes(G, pos, node_size=3, node_color=true_labels)  # 画节点
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=0.5)  # 画边
    plt.show()
def group_partition(predict_labels):
## 获取每个group的下标
    num_of_class = np.unique(predict_labels).shape[0]
    group_of_pre = []
    for i in range(num_of_class):
        temp = np.where(predict_labels == i)[0].tolist()
        group_of_pre.append(temp)
    return group_of_pre
def pplot2(node_features,embedding, name, true_labels,new_predict_labels,p):
    # embedding = np.around(embedding, 3)
    fig = plt.figure(figsize=(16,5))
    temp_labels =[0 for i in true_labels]


    ax=fig.add_subplot(131)
    ax.scatter(node_features[:, 0], node_features[:, 1], c=true_labels, s=3, cmap="rainbow",)
    tag=0
    # for a, b in zip(node_features[:, 0], node_features[:, 1]):  # 添加这个循环显示坐标
    #     plt.text(a, b, (tag), ha='center', va='bottom', fontsize=7)
    #     tag+=1
    ax.set_title("Source: Ground Truth "+name)
    plt.axis('equal')
    ax.tick_params(labelsize=8)
    ax=fig.add_subplot(132,)
    ax.scatter(embedding[:, 0], embedding[:, 1], c='grey', s=3, )
    tag = 0
    # for a, b in zip(embedding[:, 0], embedding[:, 1]):  # 添加这个循环显示坐标
    #
    #     plt.text(a, b, (tag), ha='center', va='bottom', fontsize=7)
    #     tag += 1
    # ax.set_title("Embedding of WL: Ground Truth ({})".format(name) )
    plt.axis('equal')
    ax.tick_params(labelsize=9)
    ax = fig.add_subplot(133)
    ax.scatter(embedding[:, 0], embedding[:, 1], c=true_labels, s=3, cmap="rainbow",)
    tag = 0
    # for a, b in zip(embedding[:, 0], embedding[:, 1]):  # 添加这个循环显示坐标
    #     plt.text(a, b, (a,b), ha='center', va='bottom', fontsize=10)
    #     tag += 1
    ax.set_title("Embedding: PreLlabels " + name)
    plt.axis('equal')
    ax.tick_params(labelsize=9)
    plt.show()
    plt.close()

    # ax = sns.kdeplot(white_wine['sulphates'], white_wine['alcohol'],
    #                  cmap="YlOrBr", shade=True, shade_lowest=False)
    # ax = sns.kdeplot(red_wine['sulphates'], red_wine['alcohol'],
    #                  cmap="Reds", shade=True, shade_lowest=False)

def pplot_single(embedding,true_labels,name):
    # embedding = np.around(embedding, 3)
    fig = plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    ax.scatter(embedding[:, 0], embedding[:, 1], c=true_labels, s=3, cmap="rainbow",)
    plt.axis('equal')
    ax.tick_params(labelsize=8)
    # ax.set_title(name)

    plt.show()
    plt.close()
    fig.savefig("C:/Users/Admin/Desktop/全是图片/仓库/vis_{}".format(name), bbox_inches='tight', transparent=True)
    # ax = sns.kdeplot(white_wine['sulphates'], white_wine['alcohol'],
    #                  cmap="YlOrBr", shade=True, shade_lowest=False)
    # ax = sns.kdeplot(red_wine['sulphates'], red_wine['alcohol'],
    #                  cmap="Reds", shade=True, shade_lowest=False)
def pplot3(embedding, true_labels,new_predict_labels=None):
    embedding_wl =embedding[0]
    embedding_ikwl =embedding[1]
    embedding_iikwl =embedding[2]
    temp_labels =[0 for i in true_labels]
    grid = plt.GridSpec(nrows=4, ncols=3, wspace=0.2, hspace=0.2)

    # 设置整个图像大小。
    plt.figure(figsize=(9, 8))
    # 第一个子图的具体排列位置为(0,0)。
    plt.subplot(grid[0, 0])
    plt.scatter(embedding_wl[0][:, 0], embedding_wl[0][:, 1], c=true_labels, s=2, cmap="rainbow",)
    plt.subplot(grid[1, 0])
    plt.scatter(embedding_wl[1][:, 0], embedding_wl[1][:, 1], c=true_labels, s=2, cmap="rainbow",)
    plt.subplot(grid[2, 0])
    plt.scatter(embedding_wl[2][:, 0], embedding_wl[2][:, 1], c=true_labels, s=2, cmap="rainbow",)
    plt.subplot(grid[3, 0])
    plt.scatter(embedding_wl[3][:, 0], embedding_wl[3][:, 1], c=true_labels, s=2, cmap="rainbow",)

    plt.subplot(grid[0, 1])
    plt.scatter(embedding_ikwl[0][:, 0], embedding_ikwl[0][:, 1], c=true_labels, s=2, cmap="rainbow",)
    plt.subplot(grid[1, 1])
    plt.scatter(embedding_ikwl[1][:, 0], embedding_ikwl[1][:, 1], c=true_labels, s=2, cmap="rainbow",)
    plt.subplot(grid[2, 1])
    plt.scatter(embedding_ikwl[2][:, 0], embedding_ikwl[2][:, 1], c=true_labels, s=2, cmap="rainbow",)
    plt.subplot(grid[3, 1])
    plt.scatter(embedding_ikwl[3][:, 0], embedding_ikwl[3][:, 1], c=true_labels, s=2, cmap="rainbow",)


    plt.subplot(grid[0, 2])
    plt.scatter(embedding_iikwl[0][:, 0], embedding_iikwl[0][:, 1], c=true_labels, s=2, cmap="rainbow",)
    plt.subplot(grid[1, 2])
    plt.scatter(embedding_iikwl[1][:, 0], embedding_iikwl[1][:, 1], c=true_labels, s=2, cmap="rainbow",)
    plt.subplot(grid[2, 2])
    plt.scatter(embedding_iikwl[2][:, 0], embedding_iikwl[2][:, 1], c=true_labels, s=2, cmap="rainbow",)
    plt.subplot(grid[3, 2])
    plt.scatter(embedding_iikwl[3][:, 0], embedding_iikwl[3][:, 1], c=true_labels, s=2, cmap="rainbow",)

    plt.axis('equal')
    plt.tick_params(labelsize=4)
    # ax2.tick_params(labelsize=4)
    # ax3.tick_params(labelsize=4)
    # ax4.tick_params(labelsize=4)
    # ax5.tick_params(labelsize=4)
    # ax6.tick_params(labelsize=4)
    # ax7.tick_params(labelsize=4)
    # ax8.tick_params(labelsize=4)
    # ax9.tick_params(labelsize=4)

    plt.show()
    plt.close()
    # gr''.tight_layout()
    # plt.legend()  # 让图例生效
    # plt.subplots_adjust(top=0.9, bottom=0.1, right=0.8, left=0.1, hspace=0, wspace=0)
    plt.margins(0.01, 0.01)
    # ax = sns.kdeplot(white_wine['sulphates'], white_wine['alcohol'],
    #                  cmap="YlOrBr", shade=True, shade_lowest=False)
    # ax = sns.kdeplot(red_wine['sulphates'], red_wine['alcohol'],
    #                  cmap="Reds", shade=True, shade_lowest=False)

def gsnn_adj_processing(A,r):
    lamda = r

    G = nx.from_numpy_matrix(A)
    sub_graphs = []
  # find neighbour's subgraph of each node
    for i in np.arange(len(A)):
        s_indexes = []
        s_indexes.append(i)
        for j in np.arange(len(A)):

            if (A[i][j] == 1):
                s_indexes.append(j)
        sub_graphs.append(G.subgraph(s_indexes))

    # get subgraph's nodes, adj and number of edges
    subgraph_nodes_list = []
    for i in np.arange(len(sub_graphs)):
        subgraph_nodes_list.append(list(sub_graphs[i].nodes))
    sub_graphs_adj = []
    for index in np.arange(len(sub_graphs)):
        sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())
    sub_graph_edges = []
    for index in np.arange(len(sub_graphs)):
        sub_graph_edges.append(sub_graphs[index].number_of_edges())
    # create new adj
    new_adj = torch.zeros(A.shape[0], A.shape[0])
    for node in np.arange(len(subgraph_nodes_list)):
        sub_adj = sub_graphs_adj[node]
        for neighbors in np.arange(len(subgraph_nodes_list[node])):
            index = subgraph_nodes_list[node][neighbors]
            count = torch.tensor(0).float()
            if (index == node):
                continue
            else:
                c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])
                if index in c_neighbors:
                    nodes_list = subgraph_nodes_list[node]
                    sub_graph_index = nodes_list.index(index)
                    c_neighbors_list = list(c_neighbors)
                    for i, item1 in enumerate(nodes_list):
                        if (item1 in c_neighbors):
                            for item2 in c_neighbors_list:
                                j = nodes_list.index(item2)
                                count += sub_adj[i][j]
                new_adj[node][index] = count / 2
                new_adj[node][index] = new_adj[node][index] / (len(c_neighbors) * (len(c_neighbors) - 1))
                new_adj[node][index] = new_adj[node][index] * (len(c_neighbors) ** lamda)
    weight = torch.FloatTensor(new_adj)
    weight = weight / weight.sum(1, keepdim=True)

    weight = weight + torch.FloatTensor(A)

    coeff = weight.sum(1, keepdim=True)
    coeff = torch.diag((coeff.T)[0])

    weight = weight + coeff
    weight = weight.detach().numpy()
    weight = np.nan_to_num(weight, nan=0)

    row_sum = np.array(np.sum(weight, axis=1))
    degree_matrix = np.matrix(np.diag(row_sum + 1))

    D = scipy.linalg.fractional_matrix_power(degree_matrix, -0.5)
    A_tilde_hat = D.dot(weight).dot(D)
    # A_tilde_hat = torch.FloatTensor(new_adj).detach().numpy()
    return A_tilde_hat

def GSNN(node_features, adj_mat,h):



    r=0.5
    # eps =
    # np.fill_diagonal(adj_mat,1)

    adj_mat = gsnn_adj_processing(np.array(adj_mat),r)
    # np.fill_diagonal(adj_mat, 1)
    #
    v = r * np.diag(adj_mat)
    mask = np.diag(np.ones_like(v))
    adj = mask * np.diag(v) + (1. - mask) * adj_mat
    power = np.linalg.matrix_power(adj, h)
    embedding = np.dot(power, node_features)
    return embedding



def GSNN_try(embedding, adj_mat,h):
    adj_mat = gsnn_adj_processing(np.array(adj_mat))

    power = np.linalg.matrix_power(adj_mat, h)

    embedding = math.pow(0.5,h) * (np.dot(power, embedding))
    return embedding

def adj_of_vector(adj_cur):
    # np.fill_diagonal(adj_cur, 1)
    adj_cur =np.array(adj_cur)
    deg = np.sum(adj_cur, axis=1)
    deg = np.asarray(deg).reshape(-1)
    deg[deg == 0] = 1

    deg_reverse = 1 / deg
    deg_mat = np.diag(deg_reverse)

    adj_cur = adj_cur.dot(deg_mat).T   #+adj_cur.dot(deg_mat.T).T
    # for i in range(len(adj_cur)):
    #     adj_cur[i] = adj_cur[i] / np.sum(adj_cur[i])
    deg_diag = np.diag(deg)
    adj_cur = adj_cur.dot(deg_diag)
    np.fill_diagonal(adj_cur, 1)
    adj_cur =adj_cur/np.max(adj_cur,axis=0)


    return adj_cur
import heapq


def getListMaxNumIndex(num_list, topk=3):
    num_list = num_list.tolist()
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    max_num_index = map(num_list.index, heapq.nlargest(topk, num_list))
    min_num_index = map(num_list.index, heapq.nsmallest(topk, num_list))
    max_num_index = list(max_num_index)
    min_num_index = list(min_num_index)
    return min_num_index

def WL_weighted_con(node_features, adj_mat, h):
    ####method 1
    # embedding = WL_noconcate(node_features, adj_mat, 1)
    # sim = embedding.T.dot(embedding)
    # ss =np.sum(sim,axis=1)
    # ind = getListMaxNumIndex(ss,topk=200)
    # embedding = embedding[:,ind]
    #
    # for i in range(2,h):
    #     embed = WL_noconcate(node_features, adj_mat, i)
    #
    #     embedding = np.column_stack((embedding,embed))
    #     sim = embedding.T.dot(embedding)
    #     ss = np.sum(sim, axis=1)
    #     ind = getListMaxNumIndex(ss, topk=i*200)
    #     embedding = embedding[:, ind]
    ### method2
    embedding = WL_noconcate(node_features, adj_mat, 0)


    for i in range(1, h):
        embed = WL_noconcate(node_features, adj_mat, i)
        embedding = embedding*embed


    return embedding


def knn_WL_noconcate(features_IK,node_features,adj_mat, h):
    ##获取非0下标
    k =3
    knn_filter = []
    for i in range(len(node_features)):
        neighbor_id = np.array(np.nonzero(adj_mat[i]),dtype=int).reshape(-1)
        if len(neighbor_id) <=k:
            knn_filter.append(adj_mat[i].tolist())
        else:

            temp_feat = node_features[neighbor_id]
            tree = KDTree(temp_feat, leaf_size=2)

            _, ind = tree.query([node_features[i]], k=k)
            temp_id = neighbor_id[ind]
            temp = [1 if i in temp_id else 0 for i in range(len(adj_mat))]
            knn_filter.append(temp)

    new_adj = create_adj_avg(adj_mat)
    knn_filter = np.array(knn_filter)
    new_adj = new_adj*knn_filter
    power = np.linalg.matrix_power(new_adj, h)
    embedding = math.pow(0.5,h) * (np.dot(power, features_IK))
    return embedding







def WL(node_features, adj_mat, h):
    embedding = WL_noconcate(node_features, adj_mat, 1)
    for i in range(2,h):
        embed = WL_noconcate(node_features, adj_mat, i)
        embedding = np.column_stack((embedding,embed))
    return embedding



def IGK_WL_noconcate(node_features, adj_mat, h, psi,t,):
    node_features = IK_fm_dot(node_features,t,psi)
    embedding = WL_noconcate(node_features, adj_mat, h)
    return embedding

def metric__(true_label, pred_label):
    # l1 = list(set(true_label))
    # numclass1 = len(l1)
    #
    # l2 = list(set(pred_label))
    # numclass2 = len(l2)
    # if numclass1 != numclass2:
    #     print('Class Not equal, Error!!!!')
    #     return 0
    #
    # cost = np.zeros((numclass1, numclass2), dtype=int)
    # for i, c1 in enumerate(l1):
    #     mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
    #     for j, c2 in enumerate(l2):
    #         mps_d = [i1 for i1 in mps if pred_label[i1] == c2]
    #         cost[i][j] = len(mps_d)
    # m = Munkres()
    # cost = cost.__neg__()
    # cost = cost.tolist()
    # indexes = m.compute(cost)
    #
    # new_predict = np.zeros(len(pred_label))
    # for i, c in enumerate(l1):
    #     c2 = l2[indexes[i][1]]
    #     ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
    #     new_predict[ai] = c
    #
    # nmi = normalized_mutual_info_score(true_label, pred_label)
    # acc = accuracy_score(true_label, new_predict)
    # f1 = f1_score(true_label, new_predict, average='macro')
    # return acc, nmi, f1
    pass

def IK_fm_dot(X,psi,t,):

    onepoint_matrix = np.zeros((X.shape[0], (int)(t * psi)), dtype=int)
    x_index=np.arange(len(X))
    for time in range(t):
        sample_num = psi  #
        sample_list = [p for p in range(len(X))]  # [0, 1, 2, 3]
        sample_list = random.sample(sample_list, sample_num)  # [1, 2]
        sample = X[sample_list, :]  # array([[ 4,  5,  6,  7], [ 8,  9, 10, 11]])
        # sim
        point2sample =np.dot(X,sample.T)
        min_dist_point2sample = np.argmax(point2sample, axis=1)+time*psi
       # dis
       #  from sklearn.metrics.pairwise import euclidean_distances
       #  point2sample =euclidean_distances(X,sample)
       #  min_dist_point2sample = np.argmin(point2sample, axis=1)+time*psi


        onepoint_matrix[x_index,min_dist_point2sample]=1

    return onepoint_matrix
def IK_inne_fm(X, psi, t=100):
    onepoint_matrix = np.zeros((X.shape[0], (int)(t * psi)), dtype=int)
    for time in range(t):
        sample_num = psi  #
        sample_list = [p for p in range(len(X))]
        sample_list = random.sample(sample_list, sample_num)
        sample = X[sample_list, :]

        tem1 = np.dot(np.square(X), np.ones(sample.T.shape))  # n*psi
        tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)  # n*psi

        # tem = np.dot(np.square(sample), np.ones(sample.T.shape))
        # sample2sample = tem + tem.T - 2 * np.dot(sample, sample.T)
        sample2sample = point2sample[sample_list, :]
        row, col = np.diag_indices_from(sample2sample)
        sample2sample[row, col] = 99999999
        radius_list = np.min(sample2sample, axis=1)  # 每行的最小值形成一个行向量

        min_point2sample_index = np.argmin(point2sample, axis=1)
        min_dist_point2sample = min_point2sample_index + time * psi
        point2sample_value = point2sample[range(len(onepoint_matrix)), min_point2sample_index]
        ind = point2sample_value < radius_list[min_point2sample_index]
        onepoint_matrix[ind, min_dist_point2sample[ind]] = 1
    return onepoint_matrix
from sklearn.neighbors import KDTree

from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform

def WL_noconcate(node_features, adj_mat, h):
    new_adj = create_adj_avg(adj_mat)
    np.fill_diagonal(new_adj,0.5)
    # a,b = np.linalg.eig(new_adj)

    power = np.linalg.matrix_power(new_adj, h)

    embedding = np.dot(power, node_features)
    return embedding

def WL_noconcate_gcn(node_features, adj_mat, h):
    new_adj = create_adj_avg_gcn(adj_mat)
    # new_adj = create_adj_avg(adj_mat)
    np.fill_diagonal(new_adj, 0.5)
    power = np.linalg.matrix_power(new_adj, h)

    embedding = np.dot(power, node_features)
    return embedding


def WL_noconcate_one(node_features, adj_mat):
    new_adj = create_adj_avg(adj_mat)
    np.fill_diagonal(new_adj,0.5)
    embedding = np.dot(new_adj, node_features)

    return embedding*2

def WL_noconcate_fast(node_features, adj_mat):

    embedding = np.dot(adj_mat, node_features)

    return embedding

def WL_test(node_features, adj_mat, h):
    # np.fill_diagonal(adj_mat, 1)
    deg = np.sum(adj_mat,axis=1)
    # s = pdist(node_features, 'minkowski', p=2)
    #
    # dis = squareform(s)
    # dis2 = -preprocessing.normalize(dis, norm='l1')
    # s = dis*adj_mat
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] == 1:
                adj_mat[i][j] = deg[j]

    deg = np.sum(adj_mat,axis=1)
    for i in range(adj_mat.shape[0]):
        adj_mat[i][i] = deg[i]*2

    new_adj = create_adj_avg(adj_mat)
    # G = nx.from_numpy_matrix(adj_mat)
    # neighbors_list =[list(G[i]) for i in range(G.number_of_nodes())]
    # sub_graphs = [nx.subgraph(G,neighbors) for neighbors in neighbors_list]

    # degree_by_edge = np.array([g.size() for g in sub_graphs])
    # for i in range(len(neighbors_list)):
    #     cur_list = neighbors_list[i]
    #     distance = [s[i][j] for j in cur_list]
    #     ind = cur_list[distance.index(np.max(distance))]
    #     new_adj[i][ind]= -new_adj[i][ind]

    # adjj = adj_mat
    # for i in range(3):
    #     adjj+= adjj.dot(adj_mat)
    # new_adj = adj_mat / np.sum(adj_mat, axis=1)
    # new_adj = preprocessing.normalize(new_adj, norm='l2')
    # new_adj = np.where(new_adj==0,-0.01,new_adj)
    # for i in range(new_adj.shape[0]):
    #     for j in range(i,new_adj.shape[0]):
    #         if new_adj[i][j] ==0:
    #             r = random.randint(1,100)
    #             if r>95:
    #                 new_adj[i][j] =-0.0001*r
    #                 new_adj[j][i] = -0.0001 * r

    power = np.linalg.matrix_power(new_adj, h)

    for i in range(node_features.shape[0]):
        embedding = math.pow(0.5, h) * (np.dot(power, node_features))
    # embedding = np.dot(power, node_features)
    return embedding

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj) + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def WL_gao(node_features, adj_mat, h):
    graph_feat = []
    adj = normalize_adj(adj_mat).A

    for it in range(h+1):
        if it == 0:
            graph_feat.append(node_features)
        else:
            graph_feat_cur = np.matmul(adj, graph_feat[it - 1])
            graph_feat.append(graph_feat_cur)
    adj = np.linalg.pinv(adj)
    graph_feat_last = node_features
    for it in range(h):
        graph_feat_cur = np.matmul(adj, graph_feat_last)
        graph_feat.append(graph_feat_cur)
        graph_feat_last = graph_feat_cur

    return np.concatenate(graph_feat, axis=1)



def create_adj_avg_sp(adj_mat):
    '''
    create adjacency
    '''
    adj = copy.deepcopy(adj_mat)


    deg = np.array(sp.csr_matrix.sum(adj, axis=1)).reshape(-1)
    deg = (1/ deg) * 0.5
    deg_mat = sp.diags(deg)
    adj = deg_mat*adj
    adj.setdiag(0.5)
    return adj

def adj_plot(S,Label,scale):
    n=len(Label)
    k=max(Label)+1
    plot_S = S_class_order(S, n, k, Label)
    plt.rcParams["font.size"] = 20
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("node ID", size = 13)
    ax.set_ylabel("node ID", size = 13)
    ax.spy(plot_S,markersize=.2)
    ticks = []
    for _ in range(int(n/scale)+1):
        if len(Label) > 5*scale and _ % 2 == 0:
            continue
        ticks.append(_*scale)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=8.5)

    plt.show()


def S_class_order(S, n, k, Label):
    import scipy.sparse as sp
    import random
    import copy
    partition = []
    k = max(Label)+1
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    for i in range(k):
        random.shuffle(partition[i])

    community_size = []
    for i in range(len(partition)):
        community_size.append(len(list(partition)[i]))
#     print ("community size : " + str(community_size))
    com_size_dict = {}
    for com_num, size in enumerate(community_size):
        com_size_dict[com_num] = size
    com_size_dict = dict(sorted(com_size_dict.items(), key=lambda x:x[1],  reverse=True))
#     print(com_size_dict)

    communities = copy.deepcopy(partition)
    partition = []
    for com_num in com_size_dict.keys():
        for node in list(communities)[com_num]:
               partition.append(node)
    print(len(partition))

    import random
    S_class = sp.dok_matrix((n,n))

    part_dic = {}
    for i in range(n):
        part_dic[partition[i]] = i

    nzs = S.nonzero()
    for i in range(len(nzs[0])):
        S_class[part_dic[nzs[0][i]],part_dic[nzs[1][i]]] = 1

    return S_class