# load file
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from io import open
from utilities import *
import torch
import torch.nn as nn

from Model import *
from torch import optim
import torch.nn.functional as F
import h5py
import numpy as np

import time
import math
from torch.utils.data import random_split
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_iter(input_tensor, seq_len, label, semi_label, ws, model, optimizer,  criterion_cla):
    optimizer.zero_grad()

    cla_pre = model(input_tensor, seq_len)
    if len(semi_label ==0) !=0:
        cla_loss_un = -criterion_cla(cla_pre[semi_label == 0], label[semi_label == 0] - 1) * torch.log(ws[semi_label==0])
    else:
        cla_loss_un = 0

    if len(semi_label!=0) !=0:
        cla_loss_lab = criterion_cla(cla_pre[semi_label!= 0], label[semi_label != 0] - 1)
    else:
        cla_loss_lab = 0

    cla_loss = (torch.sum(cla_loss_un) + torch.sum(cla_loss_lab))/len(label)
    cla_loss.backward()
    clip_grad_norm_(model.parameters(), 25, norm_type=2)
    optimizer.step()
    del cla_loss_lab, cla_loss_un
    return cla_loss.item(), cla_pre.detach()

def eval_iter(input_tensor, seq_len, label, model, criterion_cla):

    cla_pre = model(input_tensor, seq_len)
    cla_loss = torch.mean(criterion_cla(cla_pre, label - 1))
    return cla_loss.item(), cla_pre.detach()

def evaluation(validation_loader, k, model, criterion_cla, phase='eval'):
    #model.eval()
    seq_loss = 0
    cla_loss = 0
    pred_acc_eval = np.zeros((1, k))

    for ind, (eval_data, seq_len, label, _, _) in enumerate(validation_loader):
        input_tensor = eval_data.to(device)
        label_gt = torch.tensor(np.asarray(label), dtype=torch.long).to(device)
        loss, cla_pre = eval_iter(input_tensor, seq_len, label_gt,model,
                                         criterion_cla)

        cla_loss += loss

        pred_acc_eval = pred_acc_eval + np.asarray(topk_accuracy(cla_pre, np.asarray(label) - 1, topk=(1, 2)))


    return cla_loss / (ind + 1), pred_acc_eval[0]/(ind+1)

def training(epoch, train_loader, eval_loader,
             model, optimizer, criterion_cla,  k, file_output, network, percentage, en_num_l, hid_s, past_acc=0):
    filename = file_output.name
    auto_criterion = nn.MSELoss()
    start = time.time()
    lambda1 = lambda it: 0.95 ** (it // 150)
    model_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    cla_loss = 0
    seq_loss = 0
    global_step = 1
    pred_acc = np.zeros(k)
    for ith_epoch in range(1, epoch + 1):
        for it, (data, seq_len, label, semi_label, ws) in enumerate(train_loader):

            if ith_epoch == 1 and it == 0:

                ave_loss_train, pred_acc_train = evaluation(train_loader, k, model,criterion_cla,
                                                           )
            if global_step % 10 == 0:
                cla_loss = cla_loss / 10.0
                ave_loss_train = cla_loss
                cla_loss = 0
                seq_loss = 0

                pred_acc_train = pred_acc / 10.0
                pred_acc = np.zeros(k)
            if it % 300 == 0:

                ave_loss_eval, pred_acc_eval = evaluation(eval_loader, k, model, criterion_cla,
                                                         )
               # model.train()
                print('%s (%d %d%%)  TrainClaLoss %.4f EvalClasLoss %.4f AccTrain %.4f %.4f AccTest %.4f %.4f ' % (
                            timeSince(start, ith_epoch / epoch),
                            ith_epoch, ith_epoch / epoch * 100, ave_loss_train, ave_loss_eval,
                            pred_acc_train[0], pred_acc_train[1], pred_acc_eval[0], pred_acc_eval[1]))
                if file_output.closed:
                        file_output = open(filename, 'a')
                file_output.writelines('%.4f %.4f %.4f %.4f %.4f %.4f\n' %
                                           (ave_loss_train, ave_loss_eval,
                            pred_acc_train[0], pred_acc_train[1], pred_acc_eval[0], pred_acc_eval[1]))
                file_output.close()


            input_tensor = data.to(device)
            semi_label = torch.tensor(semi_label, dtype=torch.long).to(device)
            label = torch.tensor(label, dtype=torch.long).to(device)
            ws = torch.tensor(ws).to(device)
            total_loss, cla_pre = train_iter(input_tensor, seq_len, label, semi_label,ws, model, optimizer,
                                                     criterion_cla)
            cla_loss += total_loss

            pred_acc = pred_acc + np.asarray(topk_accuracy(cla_pre, label-1, topk=(1, 2)))[0]
            global_step += 1
            if pred_acc_eval[0] > past_acc:
                past_acc = pred_acc_eval[0]
                for item in os.listdir('./trained_model/'):
                    if item.startswith('%sAP%d_layer%d_hid%d' % (network, percentage * 100, en_num_l, hid_s)):
                        open('./trained_model/' + item, 'w').close()  # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                        os.remove('./trained_model/' + item)

                path_model = './trained_model/%sAP%d_layer%d_hid%d_epoch%d' % (network,percentage * 100, en_num_l, hid_s, ith_epoch)
                save_checkpoint(model, epoch, optimizer, ave_loss_train, path_model)
            model_scheduler.step()


    return total_loss, cla_pre







