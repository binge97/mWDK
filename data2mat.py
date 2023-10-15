import numpy as np
from sklearn.cluster import KMeans
import clustering_metric as cm
from utils import load_data
from scipy import io
from utils import  WL_noconcate,IGK_WL_noconcate,sub_wl,IK_fm_dot
import scipy.sparse as sp
import clustering_methods as cmd
import warnings
warnings.filterwarnings('ignore')




load_path = 'E:/Graph Clustering/dataset/real_world data/'
save_path = 'E:/Graph Clustering/dataset/embedding'


datasets = ['cora','citeseer','wiki','acm',]#'dblp','amap','eat','pubmed'
emb_type_li = ['wl_noc','subwl_no','ikwl']
emb_type = emb_type_li[2]
datasets = ['cora']



# for dataset in datasets:
#     adj_mat, node_features, true_labels = load_data(load_path, dataset)
#     num_of_class = np.unique(true_labels).shape[0]
#     psili=[64]
#
#     print("========================= {}:{} ============================".format(dataset, emb_type))
#     for r in range(1,2):
#         for psi in psili:
#             node_features = IK_fm_dot(node_features, 100, psi)
#             for h in range(7,8):
#                 embedding = WL_noconcate(node_features, adj_mat, h)
#                 data = {'embedding': embedding, 'class': true_labels}
#                 savepath = '{}/{}/{}_{}_psi_{}_h_{}_r_{}'.format(save_path,dataset,dataset,emb_type,psi,h,r)
#                 io.savemat('{}.mat'.format(savepath), data)
#                 print("Saving {} @psi={} h={}".format(dataset,psi,h))

dataset ='cora'
adj_mat, node_features, true_labels = load_data(load_path, dataset)
psi =64
h=7
r=1
path = '{}/{}/{}_{}_psi_{}_h_{}_r_{}.mat'.format(save_path,dataset,dataset,emb_type,psi,h,r)


data = io.loadmat(path)
embedding=data['embedding']
acc,nmi,f1,para,predict_labels = cmd.ikbc(embedding,1,7,true_labels)
print(nmi)

