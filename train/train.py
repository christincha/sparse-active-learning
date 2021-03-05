# load file
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from torch.nn.utils import clip_grad_norm_
from io import open
from utilities import *


from Model import *
from torch import optim

import numpy as np

import time
import math
from torch.utils.data import random_split
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_iter(input_tensor, seq_len, semi_ps, semi_tr, model, optimizer,  criterion_cla):
    optimizer.zero_grad()
    tr_id = semi_tr!=0
    ps_id = torch.logical_and(semi_tr==0, semi_ps!=0)
    cla_pre = model(input_tensor, seq_len)
    if sum(tr_id)!= 0:
        cla_tr = criterion_cla(cla_pre[tr_id], semi_tr[tr_id] - 1)
    else:
        cla_tr = 0

    if sum(ps_id!=0):
        cla_ps = criterion_cla(cla_pre[ps_id], semi_ps[ps_id])
    else:
        cla_ps = 0
    cla_loss = cla_tr*0.8 + 0.2* cla_ps
    cla_loss.backward()
    clip_grad_norm_(model.parameters(), 25, norm_type=2)
    optimizer.step()
    if not cla_tr == cla_ps == 0:
        cla_loss = cla_loss.item()
    return cla_loss, cla_pre

def eval_iter(input_tensor, seq_len, label, model,criterion_cla):

    cla_pre = model(input_tensor, seq_len)
    cla_loss = criterion_cla(cla_pre[label != 0], label[label != 0] - 1)
    return cla_loss.item(), cla_pre

def evaluation(validation_loader, k, model, criterion_cla, phase='eval'):
    #model.eval()
    seq_loss = 0
    cla_loss = 0
    pred_acc_eval = np.zeros((1, k))

    for ind, (eval_data, seq_len, label, semi_label,_) in enumerate(validation_loader):
        input_tensor = eval_data.to(device)
        semi_label = torch.tensor(np.asarray(label), dtype=torch.long).to(device)

        loss, cla_pre = eval_iter(input_tensor, seq_len, semi_label, model,
                                         criterion_cla)

        cla_loss += loss

        pred_acc_eval = pred_acc_eval + np.asarray(topk_accuracy(cla_pre, np.asarray(label) - 1, topk=(1, 2)))


    return cla_loss / (ind + 1), pred_acc_eval[0]/(ind+1)

def training(epoch, train_loader, eval_loader,
             model, optimizer, criterion_cla,  k, file_output, network, percentage, en_num_l, hid_s, past_acc=0):
    filename = file_output.name
    start = time.time()

    lambda1 = lambda it: 0.98 ** (it // 300)
    model_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    cla_loss = 0

    global_step = 1
    pred_acc = np.zeros(k)
    for ith_epoch in range(1, epoch + 1):
        for it, (data, seq_len, label, semi_ps, semi_tr) in enumerate(train_loader):

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
                for rp in range(3):
                    if rp==0:
                        ave_loss_eval, pred_acc_eval = evaluation(eval_loader, k, model, criterion_cla,
                                                         )
                    else:
                        _, tmp2 = evaluation(eval_loader, k, model, criterion_cla,
                                                         )
                        pred_acc_eval += tmp2
                pred_acc_eval = pred_acc_eval/3
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
            semi_ps = torch.tensor(semi_ps, dtype=torch.long).to(device)
            semi_tr = torch.tensor(semi_tr, dtype=torch.long).to(device)
            total_loss, cla_pre = train_iter(input_tensor, seq_len, semi_ps, semi_tr, model, optimizer,
                                                     criterion_cla)
            cla_loss += total_loss

            pred_acc = pred_acc + np.asarray(topk_accuracy(cla_pre, np.asarray(label)-1, topk=(1, 2)))[0]
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


    return cla_loss, cla_pre







