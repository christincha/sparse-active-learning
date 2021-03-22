
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
from ssTraining.hidden_sample import *
from  ssTraining.extract_hidden import test_extract_hidden_iter
from utility.utilities import *
from torch import optim
import torch.nn.functional as F
from  data.data_loader import *
from data.data_loader import NO_LABEL
#from  ssTraining.clustering_classification import *
import time
from  ssTraining.extract_hidden import *
class ic_train:
    def __init__(self, epoch, train_loader, eval_loader, print_every,
             model, optimizer, criterion_seq, criterion_cla, alpha, k, writer, past_acc,
             root_path, network, percentage, en_num_layers, hidden_size, labeled_bs,
             few_knn=True, TrainPS=False, T1=0, T2 = 30, af = 0.3, current_time = None):
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
        self.labeled_bs = labeled_bs
        self.root_path = root_path
        self.T1 = T1
        self.T2 = T2
        self.af = af
        self.target_num = int(np.round(len(train_loader.dataset)*percentage))
        self.semi_label = torch.tensor(train_loader.dataset.semi_label,dtype=torch.long).to(device)
        self.labeled_num = 0
        self.all_label = []
        self.select_ind = np.zeros(self.target_num)
        self.save_label = True
        self.currrent_time = current_time

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

            return cla_loss, seq_loss, en_hi, de_out.detach(), cla_pre

    def select_sample_id(self, unlab_id, cla_pre, seq_len, output):
        with torch.no_grad():
            cla_pre_trans = self.model.en_cla_forward(output,seq_len)
            pre_o = torch.softmax(cla_pre, dim=-1)
            po, ido = torch.sort(pre_o)
            pre_trans = torch.softmax(cla_pre_trans, dim=-1)
            p_re, idre = torch.sort(pre_trans)

            #dif = torch.abs(pre_o[np.arange(pre_o.shape[0]), indices] - pre_trans[np.arange(pre_o.shape[0]), indices])
            #dif = torch.sum(torch.nn.functional.kl_div(pre_trans, pre_o, reduction='none'), dim=-1)
            # vr1 = torch.argsort(dif)
            # for i in range(len(vr1)):
            #     if unlab_id[vr1[i]]:
            #         return vr1[i]
            p = random.uniform(0,1)
            if p >=0.5:
                vr1 = torch.argsort(p_re[:,-1])
                for i in range(len(vr1)):
                    if unlab_id[vr1[i]] and (ido[vr1[i], -1]!=idre[vr1[i], -1]): # vr1[i] is the 0-N batch pos
                            return vr1[i]
            else:
                vr1 = torch.argsort(po[:, -1])
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
                if self.labeled_num < self.target_num and self.epoch in [2, 5, 10,15,20]:
                    labeled_bs +=1
                    pos = self.select_sample_id(indicator, cla_pre, seq_len,de_out)
                    self.select_ind[self.labeled_num] = id[pos]
                    self.labeled_num += 1
                    self.semi_label[id[pos]] = label[pos]
                    new_cla_loss = self.cr_cla(cla_pre[pos:pos+1, :], label[pos:pos+1]) # label have been minused -1 during loading
                    total_loss = labeled_cla_loss + new_cla_loss + seq_loss
                    labeled_class.append(label[pos].cpu().item())
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

            if self.writer:
                self.writer.add_scalar('loss/'+ mode + '_step_loss_seq', seq_loss.item(), self.global_step)
                self.writer.add_scalar('acc/'+mode + '_step_accuracy_p1', acc1 / labeled_bs, self.global_step)
                self.writer.add_scalar('loss/'+ mode + '_step_loss_cla_labeled', labeled_cla_loss.item() / labeled_bs, self.global_step)
                self.writer.add_scalar('label/' + mode + 'add label', self.labeled_num,
                                       self.global_step)

            if print_freq > 0 and (it % print_freq) == 0:
                if is_train:
                    print(
                    f"[{mode}]loss[{it:<3}]\t labeled loss: {labeled_cla_loss.item():.3f}\t unlabeled loss: {new_cla_loss:.3f}\t"
                    f" loss seq: {seq_loss.item():.3f}\t Acc1: {acc1 / labeled_bs:.3%}\t labeled num {self.labeled_num:3d}"
                    )
                else:
                    print(
                        f"[{mode}]loss[{it:<3}]\t cla loss: {labeled_cla_loss.item():.3f}\t"
                        f" loss reconstruction: {seq_loss.item():.3f}\t Acc1: {acc1 / labeled_bs:.3%}\t"
                     )

        print(f">>>[{mode}]loss\t loss cla: {sum(loop_losscla)/labeled_n:.3f}\t"
              f"loss seq: {sum(loop_losskl)/len(data_loader):.3f}\t "
              f"Acc1: {sum(accuracy1) / labeled_n:.3%}")
        if self.writer:
            self.writer.add_scalar('loss/'+mode + '_epoch_loss_cla', sum(loop_losscla)/labeled_n, self.epoch)
            self.writer.add_scalar('loss/'+mode + '_epoch_loss_l1', sum(loop_losskl)/len(data_loader), self.epoch)
            self.writer.add_scalar('acc/'+ mode + '_epoch_accuracy1', sum(accuracy1) / labeled_n, self.epoch)
            self.writer.add_scalar('label/' + mode + 'epoch label', self.labeled_num,
                                   self.epoch)
            if is_train:
                if labeled_class:
                    self.all_label = self.all_label + labeled_class
                    labeled_class = np.repeat(np.asarray(labeled_class), 6)
                    self.writer.add_histogram('hist/new_labeled', np.asarray(self.all_label), self.epoch, bins='sqrt')
                    img = np.repeat(labeled_class[np.newaxis,:], 300, axis=0)
                    H, W = img.shape
                    img_HWC = np.zeros((H, W, 3))
                    img_HWC[:, :, 0] = img/60
                    img_HWC[:, :, 1] = 1 - img/60
                    self.writer.add_image('selectiong process', img_HWC, self.epoch, dataformats='HWC')
                    if self.labeled_num == self.target_num and self.save_label:
                        np.save(os.path.join('reconstruc_out/label', self.currrent_time), self.select_ind)
                        self.save_label = False

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
        self.initial_sample(train_data)
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
                self.get_class(train_data, 'train_ep%d'%ep)

    def save(self, epoch, loss=0, **kwargs):
        path = './reconstruc_out/model/'
        if not os.path.exists(path):
            os.mkdir(path)
        for item in os.listdir(path):
            if item.startswith(os.path.join(path,'%s_P%d' % (
                    self.network, self.percentage * 100))):
                open(path + item,
                     'w').close()  # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                os.remove(path + item)

        path_model = os.path.join(path,'%s_P%d_epoch%d' % (
            self.network, self.percentage * 100, epoch))
        save_checkpoint(self.model, epoch, self.optimizer, loss, path_model)

    def get_class(self, data_loader, mod):
        # 1 max prob of original input,  2. prob oftreconstruced same class as original input, # max prob of reconstructed
        with torch.no_grad():
            pall = torch.zeros((len(data_loader.dataset),3)).to(self.device)
            call = torch.zeros((len(data_loader.dataset),3)).to(self.device) # 1predict classre 2 constracut claa 3 real class
            start = 0
            for it, (data, seq_len, label, semi, id) in enumerate(data_loader):
                end = start + len(id)
                input_tensor = data.to(device)
                en_hi, de_out, cla_pre = self.model(input_tensor, seq_len)
                cla_pre_trans = self.model.en_cla_forward(de_out.detach(), seq_len)
                po = torch.softmax(cla_pre, dim=-1)
                ido = torch.sort(po)[-1][:, -1]
                pt = torch.softmax(cla_pre_trans, dim=-1)
                idt = torch.sort(pt)[-1][:, -1]
                pall[start:end, 0] = po[np.arange(len(id)), ido]
                pall[start:end, 1] = pt[np.arange(len(id)), ido]
                pall[start:end, 2] = pt[np.arange(len(id)), idt]
                call[start:end, 0] = ido
                call[start:end, 1] = idt
                call[start:end, 2] = torch.tensor(label).to(device)
                start = end
            path = './reconstruc_out/result_ana/'
            if not os.path.exists(path):
                os.mkdir(path)
        np.save(os.path.join(path, mod+'prob.py'), pall.cpu().numpy())
        np.save(os.path.join(path, mod+'class.py'), call.cpu().numpy())


    def initial_sample(self, train_loader,):
        hidden, label, semi, index = test_extract_hidden_iter(self.model, train_loader, alpha=0.5)
        train_id_list, dis_list, dis_list_prob, cluster_label  = iter_kmeans_cluster(hidden,label, index, ncluster=800)
        tmp = SampleFromCluster(train_id_list, dis_list, dis_list_prob, 'top', 0.02)
        for i in range(len(tmp)):
            self.semi_label[tmp[i]] = train_loader.dataset.label[tmp[i]]
            self.select_ind[i] = index[tmp[i]]

        self.labeled_num += len(tmp)