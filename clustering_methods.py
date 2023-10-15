import numpy as np
import sklearn.cluster as sc
import clustering_metric as cm
import scipy.sparse as sp
import warnings
from IKBC import lkbc,ik_bc
from utils import pplot2
warnings.filterwarnings('ignore')
from sklearn import preprocessing

def sc_gaussian(embedding,rep,num_of_class,true_labels):
    best_nmi,best_f1,best_acc,best_gamma= -1, -1,-1,-1
    best_nmi = -1
    best_labels = []
    # gamma_li = [0.0001, 0.001, 0.01, 0.1]
    gamma_li = [1,10,100]

    for gamma in gamma_li:
        local_acc, local_nmi, local_f1 = [], [], []
        for re in range(rep):
            predict_labels = sc.SpectralClustering(n_clusters=num_of_class, gamma=gamma).fit_predict(embedding)
            translate, new_predict_labels = cm.translate(true_labels, predict_labels)
            acc, nmi, f1, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, new_predict_labels)

            local_nmi.append(nmi)
            local_f1.append(f1)
            local_acc.append(acc)
        nmi_mean = np.mean(local_nmi)
        f1_mean = np.mean(local_f1)
        acc_mean = np.mean(local_acc)
        if best_nmi < nmi_mean:
            best_labels = new_predict_labels
            best_nmi = nmi_mean
            best_gamma = gamma
        if best_f1 < f1_mean:
            best_f1 = f1_mean
        if best_acc < acc_mean:
            best_acc = acc_mean
    para = 'sc_gamma={:.4f}'.format(best_gamma)
    return best_acc,best_nmi,best_f1,para,best_labels

def sc_linear(embedding,rep,num_of_class,true_labels):
    local_acc, local_nmi, local_f1 = [], [], []
    best_nmi =-1
    best_labels = []
    for re in range(rep):
        predict_labels = sc.SpectralClustering(n_clusters=num_of_class, affinity='linear').fit_predict(embedding)
        translate, new_predict_labels = cm.translate(true_labels, predict_labels)
        acc, nmi, f1, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, new_predict_labels)
        if nmi > best_nmi:
            best_labels = new_predict_labels
            best_nmi = nmi

        local_nmi.append(nmi)
        local_f1.append(f1)
        local_acc.append(acc)
        # pplot2(embedding, 'subgraph_wl', true_labels, new_predict_labels, 2000)

    nmi_mean = np.mean(local_nmi)
    f1_mean = np.mean(local_f1)
    acc_mean = np.mean(local_acc)
    nmi_std = np.std(local_nmi)
    f1_std = np.std(local_f1)
    acc_std = np.std(local_acc)

    return acc_mean,nmi_mean,f1_mean,'sc_linear',best_labels

def sc_semi(adj,embedding,rep,num_of_class,true_labels):
    rep = 5
    local_acc, local_nmi, local_f1 = [], [], []
    best_labels = []
    best_nmi = -1


    # embedding = preprocessing.scale(embedding)



    sim = embedding.dot(embedding.T)


    # for i in range(sim.shape[0]):
    #     for j in range(sim.shape[1]):
    #         if adj
    # sim = 0.5*adj+sim
    for re in range(rep):
        spectral = sc.SpectralClustering(n_clusters=num_of_class, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize', random_state=66)
        spectral.fit(sim)
        predict_labels = spectral.fit_predict(sim)
        translate, new_predict_labels = cm.translate(true_labels, predict_labels)
        acc, nmi, f1, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, new_predict_labels)
        if nmi > best_nmi:
            best_labels = new_predict_labels
            best_nmi = nmi
        local_nmi.append(nmi)
        local_f1.append(f1)
        local_acc.append(acc)
    nmi_mean = np.mean(local_nmi)
    f1_mean = np.mean(local_f1)
    acc_mean = np.mean(local_acc)
    nmi_std = np.std(local_nmi)
    f1_std = np.std(local_f1)
    acc_std = np.std(local_acc)
    return acc_mean, nmi_mean, f1_mean, 'sc_semi', best_labels


def km(embedding,rep,num_of_class,true_labels):

    rep=1
    local_acc, local_nmi, local_f1 = [], [], []
    best_labels = []
    best_nmi = -1
    for re in range(rep):
        predict_labels = sc.KMeans(n_clusters=num_of_class).fit_predict(embedding)
        translate, new_predict_labels = cm.translate(true_labels, predict_labels)
        acc, nmi, f1, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, new_predict_labels)
        if nmi > best_nmi:
            best_labels = new_predict_labels
            best_nmi = nmi
        local_nmi.append(nmi)
        local_f1.append(f1)
        local_acc.append(acc)
    nmi_mean = np.mean(local_nmi)
    f1_mean = np.mean(local_f1)
    acc_mean = np.mean(local_acc)
    nmi_std = np.std(local_nmi)
    f1_std = np.std(local_f1)
    acc_std = np.std(local_acc)
    return acc_mean,nmi_mean,f1_mean,'kms',best_labels


def sc_svd(embedding,rep,num_of_class,true_labels):
    rep=5
    local_acc, local_nmi, local_f1 = [], [], []
    best_labels = []
    best_nmi = -1
    u, s, v = sp.linalg.svds(embedding, k=num_of_class, which='LM')
    for re in range(rep):
        predict_labels = sc.KMeans(n_clusters=num_of_class).fit_predict(u)
        translate, new_predict_labels = cm.translate(true_labels, predict_labels)
        acc, nmi, f1, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, new_predict_labels)
        if nmi > best_nmi:
            best_labels = new_predict_labels
            best_nmi = nmi
        local_nmi.append(nmi)
        local_f1.append(f1)
        local_acc.append(acc)
    nmi_mean = np.mean(local_nmi)
    f1_mean = np.mean(local_f1)
    acc_mean = np.mean(local_acc)
    nmi_std = np.std(local_nmi)
    f1_std = np.std(local_f1)
    acc_std = np.std(local_acc)
    return acc_mean,nmi_mean,f1_mean,'sc_svd',best_labels

def ikbc(embedding,rep,num_of_class,true_labels):
    local_acc, local_nmi, local_f1 = [], [], []
    best_labels = []
    best_nmi = -1

    for re in range(rep):
        nmi_li,acc_li,f1_li=[],[],[]
        for tt in range(2, 100, 2):
            tau = tt * 0.01

            predict_labels =lkbc(embedding, num_of_class,  tau)
            translate, new_predict_labels = cm.translate(true_labels, predict_labels)

            acc, nmi, f1, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, new_predict_labels)
            if nmi >best_nmi:
                best_labels = new_predict_labels
                best_nmi = nmi
            nmi_li.append(nmi)
            f1_li.append(f1)
            acc_li.append(acc)
        nmi = np.max(nmi_li)
        acc = np.max(acc_li)
        f1 = np.max(f1_li)
        local_nmi.append(nmi)
        local_f1.append(f1)
        local_acc.append(acc)
    nmi_mean = np.mean(local_nmi)
    f1_mean = np.mean(local_f1)
    acc_mean = np.mean(local_acc)
    nmi_std = np.std(local_nmi)
    f1_std = np.std(local_f1)
    acc_std = np.std(local_acc)
    return acc_mean,nmi_mean,f1_mean,'ikbc',best_labels