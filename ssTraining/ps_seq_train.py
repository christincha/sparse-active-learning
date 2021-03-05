
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
from utilities import *
from torch import optim
import torch.nn.functional as F
from  data_loader import *
from data_loader import NO_LABEL
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

    def _iteration_step(self, input_tensor, seq_len, label, model, optimizer, criterion_seq, criterion_cla, alpha):
            optimizer.zero_grad()

            en_hi, de_out, cla_pre = model(input_tensor, seq_len)
            if sum(label != 0) != 0:
                cla_loss = criterion_cla(cla_pre, label)
            else:
                cla_loss = 0
            mask = torch.zeros([len(seq_len), max(seq_len)]).to(device)
            for ith_batch in range(len(seq_len)):
                mask[ith_batch, 0:seq_len[ith_batch]] = 1
            mask = torch.sum(mask, 1)

            seq_loss = torch.sum(criterion_seq(de_out, input_tensor), 2)
            seq_loss = torch.mean(torch.sum(seq_loss, 1) / mask)

            return cla_loss, seq_loss, en_hi, de_out, cla_pre

    def correct_label(self, unlab_id, p1,th=0.9):
        prob1 = torch.softmax(p1, dim=1)
        sort1, cla_1 = torch.sort(prob1, dim=-1)

        vr1 = sort1[ :, -1]
        meetrq = vr1 > th

        return torch.logical_and(unlab_id, meetrq)

    def _iteration(self, data_loader, print_freq, is_train=True):
        loop_losscla = []
        loop_losskl = []
        accuracy1 = []
        accuracy2 = []
        labeled_n = 0
        correct_label_epoch = 0
        mode = "train" if is_train else "test"

        for  it, (data, seq_len, label, semi_label, _) in enumerate(data_loader):
            input_tensor = data.to(device)
            semi_label = torch.tensor(semi_label, dtype=torch.long).to(device)
            label = torch.tensor(label).to(self.device)
            indicator = semi_label.eq(NO_LABEL)
            labeled_cla_loss, seq_loss, en_hi, de_out, cla_pre = self._iteration_step(input_tensor, seq_len, semi_label, self.model, self.optimizer,
                                                     self.cr_seq,
                                                     self.cr_cla, alpha=0.5)

            self.global_step += 1
            if is_train:
                labeled_bs = self.labeled_bs
                with torch.no_grad():
                    pseudo_labeled = cla_pre.max(1)[1]
                    correct_label = sum(pseudo_labeled==label)
                    correct_label_epoch += correct_label
                if self.TrainPS:
                    indicator = self.correct_label(indicator, cla_pre)
                    unlabeled_loss = self.unlabeled_weight()*torch.sum(indicator.float() *
                                               (self.cr_cla(cla_pre, pseudo_labeled)) / ((input_tensor.size(
                        0) - self.labeled_bs) / self.labeled_bs + 1e-10))
                    cla_loss = (unlabeled_loss + labeled_cla_loss)
                    unlabeled_loss = unlabeled_loss.item()
                else:
                    cla_loss = labeled_cla_loss
                    unlabeled_loss = 0
                total_loss = cla_loss + seq_loss
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), 25, norm_type=2)
                self.optimizer.step()
            else:
                labeled_bs = input_tensor.size()[0]
                unlabeled_loss = 0
                cla_loss = labeled_cla_loss/labeled_bs*self.labeled_bs
                total_loss = cla_loss + seq_loss

            labeled_n += labeled_bs

            loop_losscla.append(cla_loss.item() / len(data_loader))
            loop_losskl.append(seq_loss.item() / len(data_loader))
            acc1 = semi_label.eq(cla_pre.max(1)[1]).sum().item()
            accuracy1.append(acc1)
            if print_freq > 0 and (it % print_freq) == 0:
                if is_train:
                    print(
                    f"[{mode}]loss[{it:<3}]\t labeled loss: {labeled_cla_loss.item():.3f}\t unlabeled loss: {unlabeled_loss:.3f}\t"
                    f" loss kl: {seq_loss.item():.3f}\t Unlabeled weight: {self.unlabeled_weight():.2f} Acc1: {acc1 / labeled_bs:.3%}\t"
                    f" Correct label:{correct_label:2d}\t Per {correct_label/input_tensor.shape[0]:.3f} ")
                else:
                    print(
                        f"[{mode}]loss[{it:<3}]\t cla loss: {labeled_cla_loss.item():.3f}\t"
                        f" loss kl: {seq_loss.item():.3f}\t Unlabeled weight: {self.unlabeled_weight():.2f} Acc1: {acc1 / labeled_bs:.3%}\t"
                     )
            if self.writer:
                self.writer.add_scalar('loss/'+ mode + '_global_loss_seq', seq_loss.item(), self.global_step)
                self.writer.add_scalar('acc/'+mode + '_global_accuracy_p1', acc1 / labeled_bs, self.global_step)
                self.writer.add_scalar('loss/'+ mode + '_global_loss_cla_labeled', labeled_cla_loss / labeled_bs, self.global_step)
                if is_train:
                    self.writer.add_scalar('loss/'+ mode + '_global_loss_cla_unlabeled', unlabeled_loss, self.global_step)
                    self.writer.add_scalar('label/'+mode + '_pesudo_label_acc', correct_label, self.global_step)
                    self.writer.add_scalar(mode + '_unlabeled weight', self.unlabeled_weight(), self.global_step)

        print(f">>>[{mode}]loss\t loss cla: {sum(loop_losscla):.3f}\t"
              f"loss kl: {sum(loop_losskl):.3f}\t "
              f"Acc1: {sum(accuracy1) / labeled_n:.3%}")
        if self.writer:
            self.writer.add_scalar('loss/'+mode + '_epoch_loss_cla', sum(loop_losscla), self.epoch)
            self.writer.add_scalar('loss/'+mode + '_epoch_loss_l1', sum(loop_losskl), self.epoch)
            self.writer.add_scalar('acc/'+ mode + '_epoch_accuracy1', sum(accuracy1) / labeled_n, self.epoch)
            if is_train:
                self.writer.add_scalar(mode + '_sum of correct labeled', correct_label_epoch, self.epoch)

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
        for item in os.listdir('./seq2seq_model/'):
            if item.startswith('%s_P%d' % (
                    self.network, self.percentage * 100)):
                open('./seq2seq_model/' + item,
                     'w').close()  # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                os.remove('./seq2seq_model/' + item)

        path_model = './seq2seq_model/%s_P%d_epoch%d' % (
            self.network, self.percentage * 100, epoch)
        save_checkpoint(self.model, epoch, self.optimizer, loss, path_model)