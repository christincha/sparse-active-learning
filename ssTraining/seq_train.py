# load file
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from ssTraining.SeqModel import *
from utility.utilities import *
from torch import optim
import torch.nn.functional as F
from  data.data_loader import *

from  ssTraining.clustering_classification import *
import time
from  ssTraining.extract_hidden import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_iter(input_tensor, seq_len, label, model, optimizer, criterion_seq, criterion_cla, alpha):
    optimizer.zero_grad()
    if alpha == 0:
        en_hi, de_out = model(input_tensor, seq_len)
        cla_loss = 0
        cla_pre = None
    else:
        en_hi, de_out, cla_pre = model(input_tensor, seq_len)
        if sum(label != 0) != 0:
            cla_loss = criterion_cla(cla_pre[label != 0], label[label != 0] - 1)
        else:
            cla_loss = 0

    mask = torch.zeros([len(seq_len), max(seq_len)]).to(device)
    for ith_batch in range(len(seq_len)):
        mask[ith_batch, 0:seq_len[ith_batch]] = 1
    mask = torch.sum(mask, 1)

    seq_loss = torch.sum(criterion_seq(de_out, input_tensor), 2)
    seq_loss = torch.mean(torch.sum(seq_loss, 1) / mask)

    total_loss = alpha * cla_loss + (1 - alpha) * seq_loss

    total_loss.backward()
    clip_grad_norm_(model.parameters(), 25, norm_type=2)

    optimizer.step()
    del mask
    return total_loss, en_hi, cla_pre


def eval_iter(input_tensor, seq_len, label, model, criterion_seq, criterion_cla, alpha):
    if alpha == 0:
        # print(input_tensor.shape)
        en_hi, de_out = model(input_tensor, seq_len)

        cla_pre = None
        cla_loss = 0
    else:
        en_hi, de_out, cla_pre = model(input_tensor, seq_len)
        if sum(label != 0) > 0:
            cla_loss = criterion_cla(cla_pre[label != 0], label[label != 0] - 1)
        else:
            cla_loss = 0

    mask = torch.zeros([len(seq_len), max(seq_len)]).to(device)
    for ith_batch in range(len(seq_len)):
        mask[ith_batch, 0:seq_len[ith_batch]] = 1
    mask = torch.sum(mask, 1)

    seq_loss = torch.sum(criterion_seq(de_out, input_tensor), 2)
    seq_loss = torch.mean(torch.sum(seq_loss, 1) / mask)

    total_loss = alpha * cla_loss + (1 - alpha) * seq_loss
    del mask
    return total_loss, en_hi, cla_pre


def evaluation(validation_loader, k, model, criterion_seq, criterion_cla, alpha, phase):
    total_loss = 0
    pred_acc_eval = np.zeros((1, k))

    for ind, (eval_data, seq_len, label, semi_label,_) in enumerate(validation_loader):
        input_tensor = eval_data.to(device)
        semi_label = torch.tensor(np.asarray(semi_label), dtype=torch.long).to(device)
        if phase == 'train':
            loss, hid, cla_pred = eval_iter(input_tensor, seq_len, semi_label, model,
                                            criterion_seq, criterion_cla, alpha)
        else:
            loss, hid, cla_pred = eval_iter(input_tensor, seq_len,
                                            torch.tensor(np.asarray(label), dtype=torch.long).to(device), model,
                                            criterion_seq, criterion_cla, alpha)
        if np.isnan(loss.item()):
            print(input_tensor)
        total_loss += loss.item()
        if alpha:
            pred_acc_eval = pred_acc_eval + np.asarray(
                topk_accuracy(cla_pred, np.asarray(label) - 1, topk=(1, 2)))
    if alpha:
        pred_acc_eval = pred_acc_eval[0] / (ind + 1)
    ave_loss = total_loss / (ind + 1)
    return ave_loss, pred_acc_eval


def clustering_knn_acc(model, train_loader, eval_loader, num_class, alpha, few_knn, criterion, num_epoches=400,
                       middle_size=125):
    hi_train, hi_eval, label_train, label_eval, train_semi, eval_semi = test_extract_hidden_semi(model, train_loader,
                                                                                                 eval_loader, alpha)
    # print(hi_train.shape)

    lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 50)
    # np_out_train, np_out_eval, au_l_train, au_l_eval, au_sl_train, au_sl_eval = train_autoencoder(hi_train, hi_eval, label_train,
    #                   label_eval, train_semi, eval_semi, middle_size, criterion, lambda1, num_epoches)

    # au_pred_train, au_pred_test, au_acc_train, au_acc_test = use_kmeans_cluster(np_out_train, np_out_eval, au_l_train,
    #                                                                           au_l_eval, num_class)

    list_pred_train, list_pred_test, acc_train, acc_test = use_kmeans_cluster(hi_train, hi_eval, label_train,
                                                                              label_eval, num_class)

    if few_knn:
        hi_train, hi_eval, label_train, label_eval = few_knn_data_semi(hi_train, hi_eval,
                                                                       label_train, label_eval,
                                                                       train_semi, eval_semi)

    # print(hi_train.shape)
    knn_acc_1 = knn(hi_train, hi_eval, label_train, label_eval, nn=1)
    knn_acc_3 = knn(hi_train, hi_eval, label_train, label_eval,
                    nn=3)  # knn(np_out_train, np_out_eval, au_l_train, au_l_eval, nn=1)
    knn_acc_5 = knn(hi_train, hi_eval, label_train, label_eval, nn=9)
    return list_pred_train, list_pred_test, acc_train, acc_test, knn_acc_1, knn_acc_3, knn_acc_5


def training(epoch, train_loader, eval_loader, print_every,
             model, optimizer, criterion_seq, criterion_cla, alpha, k, file_output, past_acc,
             root_path, network, percentage, en_num_layers, hidden_size, num_class=10,
             few_knn=False, Ineration_trainning=False):
    auto_criterion = nn.MSELoss()
    start = time.time()
    lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 5)
    model_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for ith_epoch in range(1, epoch + 1):
        if ith_epoch % print_every == 0 or ith_epoch == 1:
            ave_loss_train, pred_acc_train = evaluation(train_loader, k, model, criterion_seq, criterion_cla, alpha,
                                                        phase='train')
            ave_loss_eval, pred_acc_eval = evaluation(eval_loader, k, model, criterion_seq, criterion_cla, alpha,
                                                      phase='eval')
            list_pred_train, list_pred_test, acc_train, acc_test, knn_acc_1, knn_acc_3, knn_acc_5 = clustering_knn_acc(
                model,
                train_loader,
                eval_loader,
                num_class,
                alpha,
                few_knn, criterion=auto_criterion)

            if alpha:
                print(
                    '%s (%d %d%%) TrainLoss %.4f EvalLoss %.4f TrainARI %.4f EvalARI %.4f AccTrain %.4f AccTest %.4f TrainPredk1 %.4f TrainPredk2 %.4f TestPredk1 %.4f TestPredk2 %.4f KnnACC %.4f %.4f %.4f' % (
                        timeSince(start, ith_epoch / epoch),
                        ith_epoch, ith_epoch / epoch * 100, ave_loss_train, ave_loss_eval, list_pred_train,
                        list_pred_test, acc_train, acc_test, pred_acc_train[0], pred_acc_train[1], pred_acc_eval[0],
                        pred_acc_eval[1],
                        knn_acc_1, knn_acc_3, knn_acc_5))

                file_output.writelines(
                    '%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n' %
                    (ave_loss_train, ave_loss_eval, list_pred_train,
                     list_pred_test, acc_train, acc_test, pred_acc_train[0], pred_acc_train[1],
                     pred_acc_eval[0], pred_acc_eval[1],  knn_acc_1,
                     knn_acc_3, knn_acc_5))

            else:
                print(
                    '%s (%d %d%%) TrainLoss %.4f EvalLoss %.4f TrainARI %.4f EvalARI %.4f AccTrain %.4f AccTest %.4f KnnACC %.4f %.4f %.4f' % (
                        timeSince(start, ith_epoch / epoch),
                        ith_epoch, ith_epoch / epoch * 100, ave_loss_train, ave_loss_eval, list_pred_train,
                        list_pred_test, acc_train, acc_test, knn_acc_1, knn_acc_3, knn_acc_5))
                file_output.writelines('%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n' %
                                       (ave_loss_train, ave_loss_eval, list_pred_train,
                                        list_pred_test, acc_train, acc_test, knn_acc_1,
                                        knn_acc_3, knn_acc_5))

            if ave_loss_train <  past_acc:
                past_acc = ave_loss_eval
                for item in os.listdir('./seq2seq_model/'):
                    if item.startswith('%sA%.4f_P%d_layer%d_hid%d' % (
                    network, alpha, percentage * 100, en_num_layers, hidden_size)):
                        open('./seq2seq_model/' + item,
                             'w').close()  # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                        os.remove('./seq2seq_model/' + item)

                path_model = './seq2seq_model/%sA%.4f_P%d_layer%d_hid%d_epoch%d' % (
                network, alpha, percentage * 100, en_num_layers, hidden_size, ith_epoch)
                save_checkpoint(model, epoch, optimizer, ave_loss_train, path_model)

        for it, (data, seq_len, label, semi_label,_) in enumerate(train_loader):
            input_tensor = data.to(device)
            semi_label = torch.tensor(semi_label, dtype=torch.long).to(device)
            total_loss, en_hid, cla_pre = train_iter(input_tensor, seq_len, semi_label, model, optimizer, criterion_seq,
                                                     criterion_cla, alpha)

        if ith_epoch % 50 == 0:
            filename = file_output.name
            file_output.close()
            file_output = open(filename, 'a')
        model_scheduler.step()
    return total_loss, en_hid, cla_pre


