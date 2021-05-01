import math
import numpy as np
## post processing
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import adjusted_rand_score
from sklearn import svm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import random
from scipy.spatial.distance import pdist, squareform

def iter_kmeans_cluster(train_set,  train_label,
                         train_id,  ncluster=10,
                         beta=1):
    train_label = np.asarray(train_label)

    if type(train_id) != np.ndarray:
        train_id = np.asarray(train_id)

    if type(train_label) != np.ndarray:
        train_label = np.asarray(train_label)

    kmeans = KMeans(ncluster, init='k-means++', max_iter=500, random_state=0).fit(train_set)
    pre_train = kmeans.predict(train_set)

    distance = kmeans.transform(train_set)
    distance_prob = np.exp(-beta * distance)
    distance_prob = np.divide(distance_prob, np.sum(distance_prob, 1, keepdims=True))

    DisToCenter = []
    DisToCenter_prob = []

    for i in range(len(pre_train)):
        DisToCenter.append(distance[i, pre_train[i]])
        DisToCenter_prob.append(distance_prob[i, pre_train[i]])
    DisToCenter = np.asarray(DisToCenter)
    DisToCenter_prob = np.asarray(DisToCenter_prob)

    train_id_list = []
    dis_list = []
    dis_list_prob = []
    cluster_label = np.zeros(len(train_label))
    for i in range(ncluster):
        clas_poss = pre_train == i
        cluster_label[clas_poss] = i
        train_id_list.append(train_id[clas_poss])
        dis_list.append(DisToCenter[clas_poss])
        dis_list_prob.append(DisToCenter_prob[clas_poss])
        # list_pre_test.append(Counter(corresponding_test_pre))
    return train_id_list, dis_list, DisToCenter_prob, cluster_label

def iter_kmeans_cluster_feature(train_set,  feature,
                         train_id,  ncluster=10,
                         beta=1):
    if type(train_id) != np.ndarray:
        train_id = np.asarray(train_id)
    kmeans = KMeans(ncluster, init='k-means++', max_iter=500, random_state=0).fit(train_set)
    pre_train = kmeans.predict(train_set)

    distance = kmeans.transform(train_set)

    DisToCenter = distance[range(len(pre_train)), pre_train]


    train_id_list = []
    dis_list = []
    feat_list = []
    # concate dis fea id into seperated list
    for i in range(ncluster):
        clas_poss = pre_train == i
        train_id_list.append(train_id[clas_poss])
        dis_list.append(DisToCenter[clas_poss])
        feat_list.append(feature[clas_poss])
    return train_id_list, dis_list, feat_list

def SampleFromCluster(train_id_list, dis_list, dis_list_prob, sample_method, percentage):
    # train_id_list position in one original dataset
    # dis_list: repository for distance of current sample to center
    # dis_list_pro: probility of the sample belong to one class
    # sample method: how to select samples
    # percentage: number of samples we are going to select
    num_class = len(train_id_list)
    toLabel = []
    for i in range(num_class):
        num_sample = np.round(percentage * len(dis_list[i]))

        num_sample = int(num_sample)
        # print(num_sample, len(dis_list[i]))
        if num_sample >= 1:
            # num_sample = 1

            if sample_method == 'random':
                index = train_id_list[i]
                np.random.shuffle(index)

                toLabel = toLabel + index[:num_sample].tolist()

            if sample_method == 'topbottom':
                index = train_id_list[i]
                distance = np.argsort(dis_list[i])
                num_ave = int(num_sample / 2)
                if num_ave < 1:
                    num_ave = 1
                    # toLabel = toLabel + index[distance[:num_ave]].tolist()+ index[distance[-num_ave:]].tolist()
                toLabel = toLabel + index[:num_ave].tolist() + index[-num_ave:].tolist()

            if sample_method == 'topmed':
                index = train_id_list[i]
                distance = np.argsort(dis_list[i])
                num_ave = int(num_sample / 2)
                if num_sample < 1.5:
                    num_ave = 1
                    p = random.uniform(0, 1)
                    if p > 0.5:
                        toLabel = toLabel + [index[distance[0]]]
                    else:
                        toLabel = toLabel + [index[distance[1]]]
                else:
                    interval = int(len(index) / num_sample)
                    for i in range(num_sample):
                        toLabel = toLabel + [index[distance[i * interval]]]

            if sample_method == 'top':
                index = train_id_list[i]
                distance = np.argsort(dis_list[i])
                # toLabel = toLabel + index[:num_sample].tolist()
                toLabel = toLabel + index[distance[:num_sample]].tolist()

            if sample_method == 'bottom':
                index = train_id_list[i]
                distance = np.argsort(dis_list[i])  # from smallest distance to large distance

                # toLabel = toLabel + index[:num_sample].tolist()
                toLabel = toLabel + index[distance[-num_sample:]].tolist()

            if sample_method == 'prob':
                index = train_id_list[i]
                prob = np.argsort(dis_list_prob[i])  # extract smallest probility
                toLabel = toLabel + index[prob[:num_sample]].tolist()

    return toLabel

def SampleNumber(train_id_list, dis_list, dis_list_prob, sample_method, num_sample):
    # train_id_list position in one original dataset
    # dis_list: repository for distance of current sample to center
    # dis_list_pro: probility of the sample belong to one class
    # sample method: how to select samples
    # percentage: number of samples we are going to select
    num_class = len(train_id_list)
    toLabel = []
    for i in range(num_class):

        # print(num_sample, len(dis_list[i]))
        if num_sample >= 1:
            # num_sample = 1

            if sample_method == 'random':
                index = np.concatenate(train_id_list)
                np.random.shuffle(index)

                toLabel = toLabel + index[:num_sample].tolist()
                return toLabel

            if sample_method == 'top':
                index = train_id_list[i]
                distance = np.argsort(dis_list[i])
                # toLabel = toLabel + index[:num_sample].tolist()
                toLabel.append(index[distance[0]])

            if sample_method == 'mi':
                index = train_id_list[i]
                distance = np.argsort(dis_list_prob[i])
                # toLabel = toLabel + index[:num_sample].tolist()
                toLabel.append(index[distance[0]])

            if sample_method == 'mi_prob':
                index = np.asarray(train_id_list[i])
                prob = 1 - dis_list_prob[i]
                prob = prob/np.sum(prob)
                sid = np.random.choice(index, p=prob)
                toLabel.append(sid)


    return toLabel