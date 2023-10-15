import sklearn.cluster as sc
import numpy as np
import clustering_metric as cm
from tqdm import tqdm
from utils import load_data,WL
import sys
from run import main
sys.path.append("..")
from sklearn.metrics.cluster import  normalized_mutual_info_score
import matplotlib.pyplot as plt

NMI = lambda x, y: normalized_mutual_info_score(x, y, average_method='arithmetic')

dataset = 'cora'
adj_mat, node_features, y_test, tx, ty, test_maks, true_labels = load_data(dataset)
print("**************************************DATASET: {} **************************************".format(dataset))
list_nmi, list_f1, list_acc = [], [], []
best_acc, best_nmi, best_f1,best_h,best_k= 0,0,0,0,0
# @h=11 acc:0.6971935007385525  nmi:0.5605251585747875  f1:0.6971112447102241
ac =1
hl=[11,13]
for h in hl:
    embedding = WL(node_features, adj_mat, h)

    for k in range(7,13):
        model = sc.KMeans(n_clusters=k)
        model = model.fit(embedding)
        predict_labels = model.predict(embedding)

        num_of_cluster = len(list(set(predict_labels)))
        num_of_class = len(list(set(true_labels)))
        num_of_nodes = len(true_labels)

        cluster = [[] for _ in range(num_of_cluster)] # save the index of points of each cluster
        features = [[] for i in range(num_of_cluster)]  # save the features of points of each cluster
        cluster_means= []  # get the center of each cluster
        for index in range(num_of_nodes):
            label = predict_labels[index]
            cluster[label].append(index)
            features[label].append(node_features[0][index])
        for i in range(num_of_cluster):
            index_li = cluster[i]
            em = [embedding[id] for id in index_li]
            mean = np.mean(em)
            cluster_means.append(mean)

        adj_mat1 = np.array(adj_mat[0])
        adj=[]  # save the adjacency matrix of points of each cluster
        for id in range(num_of_cluster):
            i, j = np.ix_(cluster[id],cluster[id])
            adj.append(adj_mat1[i,j])

        del_li=[]
        for i in range(num_of_cluster):
            score = main(features[i],adj[i],1,2).reshape(-1).tolist()[0]
            index_of_sort_score = np.argsort(score)
            index_li = [cluster[i][t]for t in index_of_sort_score]
            n = len(score)
            sort_score = np.zeros(n)
            for i in range(n):
                sort_score[i] = score[index_of_sort_score[i]]

            std_score = np.zeros(n)
            for i in range(n):
                if i == 0 or i == n - 1:
                    std_score[i] = np.std(sort_score)
                else:
                    std_score[i] = np.abs(np.std(sort_score[:i]) - np.std(sort_score[i:]))

            plt.plot(sort_score, std_score)
            plt.show()
            index_of_min_std = np.argsort(std_score)[0]

            del_li += index_li[:index_of_min_std]
        nn = int(len(del_li)*ac)
        del_li = del_li[:nn]
        noise_rate = round(len(del_li)*100 / num_of_nodes, 2)
        my_dict = {}
        for index, value in enumerate(true_labels):
            my_dict[index] = value

        for index in del_li:
            my_dict.pop(index)

        new_true_labels = list(my_dict.values())

        my_dict = {}
        for index, value in enumerate(predict_labels):
            my_dict[index] = value

        for index in del_li:
            my_dict.pop(index)

        new_predict_labels = list(my_dict.values())



        translate, mapping_predict_labels = cm.translate(new_true_labels, new_predict_labels)  # get mapping rules and mapped predict_labels
        normal = translate[:num_of_class]  # get the clusters of normal and noise
        noise = [i for i in range(num_of_cluster) if i not in normal]

        cluster = [[] for _ in range(num_of_cluster)]  # save the index of points of each cluster

        cluster_means = []  # get the center of each cluster
        for index in range(len(new_true_labels)):
            label = new_predict_labels[index]
            cluster[label].append(index)

        for i in range(num_of_cluster):
            index_li = cluster[i]
            em = [embedding[id] for id in index_li]
            mean = np.mean(em)
            cluster_means.append(mean)

        normal_pre = [i for i in new_predict_labels if i in normal]  # get the labels of normal and noise
        normal_true = [new_true_labels[i] for i in range(len(new_true_labels)) if new_predict_labels[i] in normal]
        nomal_means = [cluster_means[i] for i in range(num_of_cluster) if i in normal]
        min = 10000000000000

        noise_label = []
        for i in noise:
            record = -1
            for j in normal:
                temp = np.linalg.norm(cluster_means[i]-cluster_means[j])
                if temp < min:
                    record = j
            noise_label.append(record)

        final_predict_labels = []
        for p in new_predict_labels:
            if p in translate:
                final_predict_labels.append(translate.index(p))
            else:

                final_predict_labels.append(translate.index(noise_label[noise.index(p)]))






        translate, mapping_predict_labels = cm.translate(new_true_labels, final_predict_labels)  # get mapping rules and mapped predict_labels
        acc, nmi, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(new_true_labels, mapping_predict_labels)
        tqdm.write('@h:{},k:{},ACC={}, NMI={},f1_macro={}, precision_macro={}, recall_macro={}, f1_micro={}, precision_micro={}, recall_micro={},  ADJ_RAND_SCORE={}'.format(h,k,acc, nmi, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore))


        if nmi > best_nmi:
            best_nmi = nmi
            best_h = h
            best_k = k
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_h = h
            best_k = k
        if acc > best_acc:
            best_acc = acc
    print('@BEST_CLUSTER_NUMBER = {}: acc:{}  nmi:{}  f1:{}'.format(best_k,best_acc,best_nmi,best_f1))
    list_nmi.append(best_nmi)
    list_f1.append(best_f1)
    list_acc.append(best_acc)

print('@BEST_CLUSTER_NUMBER = {} h={} noise_rate:{} : acc:{}  nmi:{}  f1:{}'.format(best_k,best_h,noise_rate,best_acc,best_nmi,best_f1))
print('@mean of {} BEST_CLUSTER_NUMBER = {} h={} noise_rate:{}%: acc:{}  nmi:{}  f1:{}'.format(len(hl),best_k,best_h,noise_rate,np.mean(list_acc),np.mean(list_nmi),np.mean(list_f1)))




