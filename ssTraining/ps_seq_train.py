
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
from data.data_loader import NO_LABEL
from  ssTraining.clustering_classification import *
import time
from  ssTraining.extract_hidden import *
class ic_train:
    def __init__(self, epoch, train_loader, eval_loader, print_every,
             model, optimizer, criterion_seq, criterion_cla, alpha, k, writer, past_acc,
             root_path, network, percentage, en_num_layers, hidden_size, labeled_bs,
             few_knn=True, TrainPS=False, T1=0, T2 = 30, af = 0.3):
        self.epoch = epoch
        self.train_loader = train_loader
        self.eval_load = eval_loader
        self.model = model
        self.optimizer = optimizer
        self.cr_seq = criterion_seq
        self.cr_cla = criterion_cla
        self.k = k
        self.writer = writer
        self.network = network
        self.past_acc = past_acc
        self.percentage = percentage
        self.en_num_l = en_num_layers
        self.hid_d = hidden_size
        self.few_knn = few_knn
        self.device = device
        self.global_step = 0
        self.TrainPS = TrainPS
        self.labeled_bs = labeled_bs
        self.root_path = root_path
        self.T1 = T1
        self.T2 = T2
        self.af = af
        self.traget_num = np.round(len(train_loader))
        self.semi_label = torch.tensor(train_loader.dataset.semi_label,dtype=torch.long).to(device)

    def _iteration_step(self, input_tensor, seq_len, label, model, optimizer, criterion_seq, criterion_cla, alpha):
            optimizer.zero_grad()

            en_hi, de_out, cla_pre = model(input_tensor, seq_len)

            cla_loss = criterion_cla(cla_pre, label)
            mask = torch.zeros([len(seq_len), max(seq_len)]).to(device)
            for ith_batch in range(len(seq_len)):
                mask[ith_batch, 0:seq_len[ith_batch]] = 1
            mask = torch.sum(mask, 1)

            seq_loss = torch.sum(criterion_seq(de_out, input_tensor), 2)
            seq_loss = torch.mean(torch.sum(seq_loss, 1) / mask)

            return cla_loss, seq_loss, en_hi, de_out.detach, cla_pre

    def select_sample_id(self, unlab_id, p1):
        prob1 = torch.softmax(p1, dim=1) #(N, 60)
        sort1, cla_1 = torch.sort(prob1, dim=-1)

        vr1 = torch.argsort(sort1[ :, -1])

        for i in range(len(vr1)):
            if unlab_id[vr1[i]]:
                return vr1[i]

    def _iteration(self, data_loader, print_freq, is_train=True):
        loop_losscla = []
        loop_losskl = []
        accuracy1 = []
        labeled_n = 0
        correct_label_epoch = 0
        labeled_class = []
        mode = "train" if is_train else "test"

        for  it, (data, seq_len, label, semi, id) in enumerate(data_loader):
            input_tensor = data.to(device)
            if is_train:
                semi= self.semi_label[id]
            else:
                semi = torch.tensor(semi, dtype=torch.long).to(self.device)
            label = torch.tensor(label).to(self.device)
            indicator = semi.eq(NO_LABEL)
            labeled_bs = len(semi)-sum(indicator)
            labeled_cla_loss, seq_loss, en_hi, de_out, cla_pre = self._iteration_step(input_tensor, seq_len, semi, self.model, self.optimizer,
                                                     self.cr_seq,
                                                     self.cr_cla, alpha=0.5)

            self.global_step += 1
            if is_train:
                if self.global_step <= self.traget_num:
                    labeled_bs +=1
                    pos = self.select_sample_id(indicator, cla_pre)
                    self.semi_label[id[pos]] = label[pos]
                    new_cla_loss = self.cr_cla(cla_pre[pos:pos+1, :], label[pos:pos+1]-1)
                    total_loss = labeled_cla_loss + new_cla_loss + seq_loss
                    labeled_class.append(label[pos])
                    new_cla_loss = new_cla_loss.detach().item()
                    seq_loss.detach()

                else:
                    total_loss = labeled_cla_loss + seq_loss
                    new_cla_loss = 0
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), 25, norm_type=2)
                self.optimizer.step()
            else:
                labeled_bs = input_tensor.size()[0]

            labeled_n += labeled_bs
            loop_losscla.append(labeled_cla_loss.item())
            loop_losskl.append(seq_loss.item())
            acc1 = semi.eq(cla_pre.max(1)[1]).sum().item()

            accuracy1.append(acc1)
            if print_freq > 0 and (it % print_freq) == 0:
                if is_train:
                    print(
                    f"[{mode}]loss[{it:<3}]\t labeled loss: {labeled_cla_loss.item():.3f}\t unlabeled loss: {new_cla_loss:.3f}\t"
                    f" loss seq: {seq_loss.item():.3f}\t Acc1: {acc1 / labeled_bs:.3%}\t"
                    )
                else:
                    print(
                        f"[{mode}]loss[{it:<3}]\t cla loss: {labeled_cla_loss.item():.3f}\t"
                        f" loss reconstruction: {seq_loss.item():.3f}\t Acc1: {acc1 / labeled_bs:.3%}\t"
                     )
            if self.writer:
                self.writer.add_scalar('loss/'+ mode + '_step_loss_seq', seq_loss.item(), self.global_step)
                self.writer.add_scalar('acc/'+mode + '_step_accuracy_p1', acc1 / labeled_bs, self.global_step)
                self.writer.add_scalar('loss/'+ mode + '_step_loss_cla_labeled', labeled_cla_loss.item() / labeled_bs, self.global_step)

        print(f">>>[{mode}]loss\t loss cla: {sum(loop_losscla)/labeled_n:.3f}\t"
              f"loss seq: {sum(loop_losskl)/len(data_loader):.3f}\t "
              f"Acc1: {sum(accuracy1) / labeled_n:.3%}")
        if self.writer:
            self.writer.add_scalar('loss/'+mode + '_epoch_loss_cla', sum(loop_losscla)/labeled_n, self.epoch)
            self.writer.add_scalar('loss/'+mode + '_epoch_loss_l1', sum(loop_losskl)/len(data_loader), self.epoch)
            self.writer.add_scalar('acc/'+ mode + '_epoch_accuracy1', sum(accuracy1) / labeled_n, self.epoch)
            if is_train:
                if labeled_class:
                    self.writer.add_histogram('hist/new_labeled', np.asarray(labeled_class), self.epoch)

    def unlabeled_weight(self):
        alpha = 0.0
        if self.epoch > self.T1:
            alpha = (self.epoch - self.T1) / (self.T2 - self.T1) * self.af
            if self.epoch > self.T2:
                alpha = self.af
        return alpha

    def train(self, data_loader, print_freq=20):
        self.model.train()
        with torch.enable_grad():
            self._iteration(data_loader, print_freq)

    def test(self, data_loader, print_freq=10):
        self.model.eval()
        with torch.no_grad():
            self._iteration(data_loader, print_freq, is_train=False)


    def loop(self, epochs, train_data, test_data, scheduler=None, print_freq=-1, save_freq=1):
        for ep in range(epochs):
            self.epoch = ep
            print("------ Training epochs: {} ------".format(ep))
            self.train(train_data, print_freq)
            if scheduler is not None:
                scheduler.step()
            print("------ Testing epochs: {} ------".format(ep))
            self.test(test_data, print_freq)
            if ep % save_freq == 0:
                self.save(ep)

    def save(self, epoch, loss=0, **kwargs):
        for item in os.listdir('../seq2seq_model/'):
            if item.startswith('%s_P%d' % (
                    self.network, self.percentage * 100)):
                open('../seq2seq_model/' + item,
                     'w').close()  # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                os.remove('../seq2seq_model/' + item)

        path_model = '../seq2seq_model/%s_P%d_epoch%d' % (
            self.network, self.percentage * 100, epoch)
        save_checkpoint(self.model, epoch, self.optimizer, loss, path_model)