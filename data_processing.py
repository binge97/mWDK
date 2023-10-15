import numpy as np
from scipy import io
import linecache
import pickle as pkl
import scipy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch
import pickle
import warnings
warnings.filterwarnings('ignore')
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def data2npy(dataset):
    print('Processing Dataset:  {}'.format(dataset))

    if dataset in ["amazon_cs"]:
        data = np.load("./dataset/real_world data/raw_data/{}/{}.npz".format(dataset,dataset))
        true_labels = data['labels']
        adj= data["adj_data"]
        adj_indices =data["adj_indices"]
        adj_indptr = data["adj_indptr"]
        adj_shape = data['adj_shape']
        attr = data["attr_data"]
        attr_indices =data["attr_indices"]
        attr_indptr = data["attr_indptr"]
        attr_shape = data["attr_shape"]

        adj_mat = np.array(sp.csr_matrix((adj, adj_indices, adj_indptr), shape=adj_shape).todense(),dtype=float)
        node_features = np.array(sp.csr_matrix((attr, attr_indices, attr_indptr), shape=attr_shape).todense(),dtype=float)

    if dataset in ["acm"]:


        # dataset = 'wiki'
        data = sio.loadmat('./dataset/real_world data/raw_data/{}/{}.mat'.format(dataset,dataset))
        node_features = data['fea']
        if sp.issparse( node_features):
            node_features = node_features.todense()

        adj_mat = data['W']
        adj_mat =np.where(adj_mat>0,1,0)
        gnd = data['gnd']
        gnd = gnd.T
        gnd = gnd - 1
        true_labels = np.array(gnd[0, :])


    if dataset in ['dblp']:
            ## Load data
            data = sio.loadmat('./dataset/real_world data/raw_data/{}/{}.mat'.format(dataset,'DBLP4057_GAT_with_idx'))

            X = data['features']
            A = data['net_APTPA']
            B = data['net_APCPA']
            C = data['net_APA']
            # D = data['PTP']—

            if sp.issparse(X):
                X = X.todense()
            # X_ = []
            # X_.append(np.array(X))
            adj_mat = []
            print(np.sum(A),np.sum(B),np.sum(C))
            adj_mat.append(A)
            adj_mat.append(B)
            adj_mat.append(C)
            # X_.append(np.array(A))
            # X_.append(np.array(B))
            # X_.append(np.array(C))
            # av.append(C)
            # av.append(D)
            gnd = data['label']
            gnd = gnd.T
            gnd = np.argmax(gnd, axis=0)
            true_labels = np.array(gnd)
            node_features=X

    if dataset in ['wiki']:
        print('@data by "AGC"')

        # dataset = 'wiki'
        data = sio.loadmat('./dataset/real_world data/raw_data/{}/{}.mat'.format(dataset,dataset))
        node_features = data['fea']
        adj_mat = data['W']
        if sp.issparse( node_features):
            node_features = node_features.todense()
        if sp.issparse(adj_mat):
            adj_mat = adj_mat.todense()
        # adj_mat = np.where(adj_mat>0,1,0)
        node_features = np.where(node_features > 0, 1.0, 0.0)
        gnd = data['gnd']
        gnd = gnd.T
        gnd = gnd - np.min(gnd)
        true_labels = np.array(gnd[0, :])



    if dataset in ['cora','citeseer','pubmed']:




        print("data by 'original'")
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            '''
            fix Pickle incompatibility of numpy arrays between Python 2 and 3
            https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
            '''
            with open("./dataset/real_world data/raw_data/{}/ind.{}.{}".format(dataset,dataset, names[i]), 'rb') as rf:
                u = pkl._Unpickler(rf)
                u.encoding = 'latin1'
                cur_data = u.load()
                objects.append(cur_data)
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file( "./dataset/real_world data/raw_data/{}/ind.{}.test.index".format(dataset,dataset))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        # idx_test = test_idx_range.tolist()
        # idx_train = range(len(y))
        # idx_val = range(len(y), len(y) + 500)

        # train_mask = sample_mask(idx_train, labels.shape[0])
        # val_mask = sample_mask(idx_val, labels.shape[0])
        # test_mask = sample_mask(idx_test, labels.shape[0])
        #
        # y_train = np.zeros(labels.shape)
        # y_val = np.zeros(labels.shape)
        # y_test = np.zeros(labels.shape)
        # y_train[train_mask, :] = labels[train_mask, :]
        # y_val[val_mask, :] = labels[val_mask, :]
        # y_test[test_mask, :] = labels[test_mask, :]
        node_features = np.array(features.todense())
        adj_mat = np.array(adj.todense())
        true_labels = np.argmax(labels, 1)

    adj_mat = adj_mat.astype(float)
    np.fill_diagonal(adj_mat,0)
    node_features = node_features.astype(float)
    np.save('./dataset/real_world data/raw_data/{}/{}_adj'.format(dataset,dataset), adj_mat)
    np.save('./dataset/real_world data/raw_data/{}/{}_feat'.format(dataset,dataset) , node_features)
    np.save('./dataset/real_world data/raw_data/{}/{}_label'.format(dataset,dataset), true_labels)
    return adj_mat, node_features, true_labels


def load_graph_data(path,dataset_name):

    """
    load graph data
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :return: the features, labels and adj
    """
    load_path = "{}/raw_data/{}/{}".format(path,dataset_name,dataset_name)

    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)

    details ="++++++++++++++++++++++++++++++++\n--DETAILS OF GRAPH: [{}]--\n++++++++++++++++++++++++++++++++\n" \
             "+ dataset name:   {} \n+ feature shape:  {} \n+ label shape:    {} \n+ adj shape:      {} \n" \
             "+ edge num:   {} \n+ category num:       {} \n+ category distribution:  \n".format(dataset_name,dataset_name,feat.shape,label.shape,adj.shape,int(adj.sum()/2),max(label)-min(label)+1)
    for i in range(max(label)+1):
        details+="+ label {}: {}\n".format(i,len(label[np.where(label == i)]))
    details+="++++++++++++++++++++++++++++++++\n"

    print(details)
    return feat, label, adj, details




if __name__ == '__main__':
    # datasets = ['cora', 'citeseer', 'pubmed', 'wiki','acm','flickr', 'blogcatalog']
    # datasets = ['Graph_1','Graph_2','Graph_3','Graph_4','Graph_5','Graph_6',]
    datasets = ['inituition']
    # datasets =['ENodes_UDegrees']

    # data processing to npy
    # for dataset in datasets:
    #     adj_mat, node_features, true_labels = data2npy(dataset)


    ## npy to mat
    path ='E:/Graph Clustering/dataset/artificial data'
    # path = 'E:/Graph Clustering/dataset/real_world data'
    for dataset in datasets:
        features, labels, adj_mat, details = load_graph_data(path,dataset)

        data={'features':features, 'labels':labels, 'adj_mat':adj_mat, 'details':details}

        io.savemat('{}/{}.mat'.format(path,dataset),data)
    print("Saving End！")