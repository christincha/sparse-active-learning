import numpy as np
import os
from collections import Counter
from data.data_loader import *
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import ConfusionMatrixDisplay
root_path = './labels'
from scipy import stats

def summary_label(threshold, semi_name, rep = 5):
    for drop in [0.1]:
        semi_label = np.load(semi_name)
        labels = np.zeros([len(semi_label), rep])
        for ith_rep in range(rep):
            label_name = 'ps_prob_ith%d_drop%.2f.npy' %(ith_rep, drop)
            tmp = np.load(os.path.join(root_path, label_name))
            lab_tmp = np.argmax(tmp, axis=1) + 1
            labels[:, ith_rep] = lab_tmp
        lab, counts = stats.mode(labels, axis=1)
        index = np.logical_and(semi_label==0, counts[:,0] >3)
        semi_label[index] = lab[index,0]
        np.save('label_part3_drop%.2f' %drop, semi_label)

def summary_label_prob(threshold, semi_name, prob, rep = 5):
    for drop in [0.3]:
        semi_label = np.load(semi_name)
        max_arg = np.argmax(prob, axis=-1)
        lab, counts = stats.mode(max_arg, axis=0)
        index = np.logical_and(semi_label==0, counts[0] >29)
        semi_label[index] = lab[0][index]+1
    return semi_label, sum(index)

def average_label_prob(threshold, semi_name, prob, rep = 5):
    for drop in [0.3]:
        semi_label = np.load(semi_name)
        ave_prob = np.average(prob, axis=0)
        max_arg = np.argmax(ave_prob, axis=-1)
        max_prob = np.max(ave_prob, axis=-1)
        index = np.logical_and(semi_label==0,max_prob> threshold )
        semi_label[index] = max_arg[index]+1
    return semi_label, sum(index)

def average_label_base(threshold, semi_name, prob, true_label, rep = 5):
    for drop in [0.3]:
        true_label = np.asarray(true_label)
        semi_label = np.load(semi_name)
        ave_prob = np.average(prob, axis=0)
        max_arg = np.argmax(ave_prob, axis=-1)
        max_prob = np.max(ave_prob, axis=-1)
        index = np.logical_and(semi_label==0,max_prob> threshold )
        index = np.logical_and(index, max_arg+1 == true_label)
        semi_label[index] = max_arg[index]+1
    return semi_label, sum(index)

def variance_ratio_label(threshold, semi_name, prob, rep = 5):
    for drop in [0.3]:
        semi_label = np.load(semi_name)
        ave_prob = np.average(prob, axis=0)
        max_arg = np.argmax(ave_prob, axis=-1)
        sort_prob = np.sort(ave_prob, axis=-1)
        vr = sort_prob[:,-1] - sort_prob[:,-2]
        index = np.logical_and(semi_label==0,vr > threshold )
        semi_label[index] = max_arg[index]+1
    return semi_label, sum(index)

def analysis_label(label, semi_label, pred_label, num_ps, prob):
    if type(label) is not np.ndarray:
        label = np.asarray(label)

    if type(semi_label) is not np.ndarray:
        semi_label = np.asarray(semi_label)
    prob = np.mean(prob, axis=0)
    selected = semi_label != pred_label
    assert sum(selected) == num_ps
    true_label = label[selected]
    ps_label = pred_label[selected]
    cla_prob = prob[selected, :]
    num_correct = sum(true_label==ps_label)
    summary = {}
    for i in range(1, 61):
        id = true_label==i
        summary[str(i)]= {}
        summary[str(i)]['Num True'] = sum(id)
        summary[str(i)]['Pred Stat'] = Counter(ps_label[id])

    return summary, true_label, ps_label, cla_prob

def mc_dropout_evaluate(prob_name, base_semi, classes):
    labeled = np.load(base_semi)
    semi_all = np.load(prob_name)
    x_len = sum(labeled==0)
    acc = None
    #compute mean
    y_T = semi_all[:, labeled==0, :]
    y_mean = np.mean(y_T, axis=0)
    assert y_mean.shape == (x_len, classes)

    #compute majority prediction
    y_pred = np.array([np.argmax(np.bincount(row)) for row in np.transpose(np.argmax(y_T, axis=-1))])+1
    assert y_pred.shape == (x_len,)

    #compute variance
    y_var = np.var(y_T, axis=0)
    assert y_var.shape == (x_len, classes)

    return y_mean, y_var, y_pred, y_T

if __name__ == '__main__':
    drop = 0.
    ProjectFolderName = 'NTUProject'
    root_path = '/home/ws2/Documents/jingyuan/'
    train_data = 'NTUtrain_cs_full.h5'
    dataset_train = MySemiDataset(os.path.join(root_path, ProjectFolderName, train_data), 0)
    prob = np.load('./labels/SSLBase_ps_prob_full_rp1_drop%.2f.npy' % (drop))
    semi_name = './labels/base_semiLabel.npy'
    pred, len_ps = average_label_base(0.3, semi_name, prob, dataset_train.label, rep=5)
    #summary_label(threshold, './labels/base_semiLabel.npy')
    summary, true_label, ps_label, cla_prob = analysis_label(dataset_train.label, np.load(semi_name), pred, len_ps,prob)
    con_matrix = confusion_matrix(true_label, ps_label, normalize='true')
    fig = plt.figure(1, figsize=(100, 100), dpi=300)
    font = {'family': 'normal',
            'size': 2}

    matplotlib.rc('font', **font)
    # your confusion matrix
    ls = range(60)  # [0, 1] # your y labels
    disp = ConfusionMatrixDisplay(confusion_matrix=con_matrix, display_labels=ls)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
    plt.savefig('./pslabel_p5_A1.jpg')
    plt.show()
    for i in range(60):

        mis_cla = np.where(con_matrix[i] > 0.1)[0]
        if len(mis_cla) > 0:
            print('class%d' % i, 'mis_classified as', mis_cla, 'prob', con_matrix[i][mis_cla])
            if len(mis_cla) > 2:
                pos = np.logical_and(true_label==i+1, ps_label!=i+1)
                tmp_prob = np.mean(cla_prob[pos,:][:, mis_cla], axis=0)
                print('mis prob', tmp_prob)