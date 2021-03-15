
import os
from utility.utilities import *
from torch import optim
from data.relic_Dataset import NO_LABEL
from pathlib import Path
import torch.nn as nn
import random
from copy import deepcopy
def nan_to_num(input):
    if torch.isnan(input):
        return 0
    else:
        return input

class relic_train_copy:
    def __init__(self, epoch, train_loader, eval_loader,
                 model, optimizer, cr_cla, cr_kl, k, writer,
                 network, device, T1, T2, af, labeled_bs, past_acc=0.3, percentage=0.05, en_num_l=3, hid_s=1024, few_knn=True, current_time= None):
        self.epoch = epoch
        self.train_loader = train_loader
        self.eval_load = eval_loader
        self.model = model
        #self.model_copy = deepcopy(model)
        self.optimizer = optimizer
        self.cr_cla = cr_cla
        self.cr_kl = cr_kl
        self.k = k
        self.writer = writer
        self.network = network
        self.past_acc = past_acc
        self.percentage = percentage
        self.en_num_l = en_num_l
        self.hid_d = hid_s
        self.few_knn = few_knn
        self.device = device
        self.global_step = 0
        self.T1 = T1
        self.T2 = T2
        self.af = af
        self.labeled_bs = labeled_bs
        self.semi_label = torch.tensor(train_loader.dataset.semi_label,dtype=torch.long).to(device)
        self.labeled_num = len(self.semi_label) - sum(self.semi_label==NO_LABEL)
        self.target_num = np.int(np.round(len(self.semi_label)*percentage))
        self.select_ind = np.zeros(self.target_num)
        self.current_time = current_time
        self.all_label = []
        self.save_label = True

    def correct_label(self, unlab_id, p1, p2,th=0.5):
        prob1 = torch.exp(p1)
        prob2 = torch.exp(p2)
        sort1, cla_1 = torch.sort(prob1, dim=-1)
        sort2, cla_2 = torch.sort(prob2, dim=-1)
        vr1 = sort1[ :, -1]
        vr2 = sort2[:, -1]

        meetrq = torch.logical_and(vr1 > th, vr2 > th)

        return torch.logical_and(unlab_id, meetrq)

    def select_sample_id(self, unlab_id,p1,p2):

        prob1 = torch.exp(p1)
        prob2 = torch.exp(p2)
        sort1, cla_1 = torch.sort(prob1, dim=-1)
        sort2, cla_2 = torch.sort(prob2, dim=-1)
        #dif = torch.abs(sort1[:,-1]- prob2[np.arange(sort2.shape[0]), cla_2[:,-1]])
        #vr1 = list(range(p1.shape[0]))#torch.random.shuffle(torch.arange(len(dif)))  #torch.argsort(dif)
        #random.shuffle(vr1)
        vr1 = torch.argsort(sort2[:,-1])
        for i in range(len(vr1)):
            # if unlab_id[vr1[-i]] and cla_2[vr1[-i],-1] != cla_1[vr1[-i], -1]:
            #     return vr1[-i]
            if unlab_id[vr1[i]] and cla_1[vr1[i] , -1]!=cla_2[vr1[i], -1]:
                return  vr1[i]
        for i in range(len(vr1)):
            # if unlab_id[vr1[-i]] and cla_2[vr1[-i],-1] != cla_1[vr1[-i], -1]:
            #     return vr1[-i]
            if unlab_id[vr1[i]]:
                return  vr1[i]

    def _iteration(self, data_loader, print_freq, is_train=True):
        loop_losscla = []
        loop_losskl = []
        accuracy1 = []
        accuracy2 = []
        labeled_class=[]
        labeled_n = 0
        mode = "train" if is_train else "test"

        for it, (in1, in2, len1, len2, label, semi, idx) in enumerate(data_loader):
            in1 = in1.to(self.device)
            in2 = in2.to(self.device)
            #change to original label to debug
            semi = torch.tensor(semi, dtype=torch.long).to(self.device)
            label = torch.tensor(label, dtype=torch.long).to(self.device)
            self.global_step += 1
            # compute output from old model and also current model
            enhi1, p1, enhi2, p2 = self.model(in1, in2, len1, len2)
            ##_,p1_old, _,p2_old = self.model_copy(in1, in2, len1, len2)

            if is_train:
                semi = self.semi_label[idx]
                indicator = semi.eq(NO_LABEL)
                labeled_loss = torch.sum(self.cr_cla(p1, semi) + self.cr_cla(p2, semi))
                labeled_bs = len(indicator) - sum(indicator)

                if self.labeled_num < self.target_num and self.epoch%5==0:
                    pos = self.select_sample_id(indicator, p1, p2)
                    self.select_ind[self.labeled_num] = idx[pos]
                    self.labeled_num +=1
                    self.semi_label[idx[pos]] = label[pos]
                    labeled_class.append(label[pos])
                    new_loss = torch.sum(self.cr_cla(p1[pos:pos+1],label[pos:pos+1]) + self.cr_cla(p2[pos:pos+1], label[pos:pos+1]))
                    labeled_n += 1
                    loss_cla = new_loss + labeled_loss
                    new_loss = new_loss.item()
                else:
                    new_loss = 0
                    loss_cla = labeled_loss

                loss_kl = self.cr_kl(p1[indicator], p2[indicator]) / 2 + self.cr_kl(p2[indicator], p1[indicator]) / 2

                loss = loss_kl + loss_cla
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                semi = torch.tensor(semi, dtype=torch.long).to(self.device)
                labeled_bs = in1.size()[0]
                loss_cla = torch.sum(self.cr_cla(p1, label) + self.cr_cla(p2, label))/in1.size()[0]
                labeled_loss = loss_cla
                unlabeled_loss = 0
                loss_kl = self.cr_kl(p1, p2) / 2 + self.cr_kl(p2, p1) / 2
                loss = loss_cla + loss_kl

            labeled_n += labeled_bs

            loop_losscla.append(loss_cla.item() / len(data_loader))
            loop_losskl.append(loss_kl.item() / len(data_loader))
            acc1 = semi.eq(p1.max(1)[1]).sum().item()
            acc2 = semi.eq(p2.max(1)[1]).sum().item()
            accuracy1.append(acc1)
            accuracy2.append(acc2)
            if print_freq > 0 and (it % print_freq) == 0:
                if is_train:
                    print(
                    f"[{mode}]loss[{it:<3}]\t labeled loss: {labeled_loss.item():.3f}\t new loss: {new_loss:.3f}\t"
                    f" loss kl: {loss_kl.item():.3f}\t  Acc1: {nan_to_num(acc1 / labeled_bs):.3%}\t"
                    f"Acc2 : {nan_to_num(acc2 / labeled_bs):.3%} labeled_num:{self.labeled_num:.1f} ")
                else:
                    print(
                        f"[{mode}]loss[{it:<3}]\t cla loss: {labeled_loss.item():.3f}\t"
                        f" loss kl: {loss_kl.item():.3f}\t Acc1: {acc1 / labeled_bs:.3%}\t"
                        f"Acc2 : {acc1 / labeled_bs:.3%} ")
            if self.writer:
                self.writer.add_scalar('loss/'+mode + '_global_loss_cla', loss_cla.item(), self.global_step)
                self.writer.add_scalar('loss/'+mode + '_global_loss_kl', loss_kl.item(), self.global_step)
                self.writer.add_scalar('acc/'+mode + '_global_accuracy_p1', acc1 / labeled_bs, self.global_step)
                self.writer.add_scalar('acc/'+mode + '_global_accuracy_p2', acc2 / labeled_bs, self.global_step)

        print(f">>>[{mode}]loss\t loss cla: {sum(loop_losscla):.3f}\t"
              f"loss kl: {sum(loop_losskl):.3f}\t "
              f"Acc1: {sum(accuracy1) / labeled_n:.3%}, Acc1: {sum(accuracy2) / labeled_n:.3%}")
        if self.writer:
            self.writer.add_scalar('global_loss/'+mode + '_epoch_loss_cla', sum(loop_losscla), self.epoch)
            self.writer.add_scalar('global_loss/'+mode + '_epoch_loss_kl', sum(loop_losskl), self.epoch)
            self.writer.add_scalar('global_acc/'+mode + '_epoch_accuracy1', sum(accuracy1) / labeled_n, self.epoch)
            self.writer.add_scalar('global_acc/'+mode + '_epoch_accuracy2', sum(accuracy2) / labeled_n, self.epoch)
            if is_train:
                if labeled_class:
                    self.all_label = self.all_label + labeled_class
                    labeled_class = np.repeat(np.asarray(labeled_class), 6)
                    self.writer.add_histogram('hist/new_labeled', np.asarray(self.all_label), self.epoch, bins='sqrt')
                    img = np.repeat(labeled_class[np.newaxis, :], 300, axis=0)
                    H, W = img.shape
                    img_HWC = np.zeros((H, W, 3))
                    img_HWC[:, :, 0] = img / 60
                    img_HWC[:, :, 1] = 1 - img / 60
                    self.writer.add_image('selectiong process %d' % self.epoch, img_HWC, self.epoch, dataformats='HWC')
                    if len(self.all_label) == self.target_num and self.save_label:
                        np.save(os.path.join('relic_out/label', self.current_time), self.select_ind)
                        self.save_label = False
    def unlabeled_weight(self):
        alpha = 0
        if self.epoch > self.T1:
            alpha = (self.epoch - self.T1) / (self.T2 - self.T1) * self.af
            if self.epoch > self.T2:
                alpha =  self.af

        return alpha

    def train(self, data_loader, print_freq=20):
        self.model.train()
        with torch.enable_grad():
            self._iteration(data_loader, print_freq)

    def test(self, data_loader, print_freq=10):
        self.model.eval()
        with torch.no_grad():
            self._iteration(data_loader, print_freq, is_train=False)

    def update_parameter(self, tao=0.8):
        # mp = list(self.model.parameters())
        # mpc = list(self.model_copy.parameters())
        # n = len(mp)
        # for i in range(0, n):
        #     mpc[i].data[:] = tao*mpc[i].data[:] + (1-tao)*mp[i].data[:]
        del self.model_copy
        self.model_copy = deepcopy(self.model)

    def loop(self, epochs, train_data, test_data, scheduler=None, print_freq=-1, save_freq=1):
        for ep in range(epochs):
            self.epoch = ep
            if ep == 0:
                self.save(ep)
            print("------ Training epochs: {} ------".format(ep))
            # if (ep)%50==0:
            #     self.update_parameter()
            #     self.re_initialize()
            self.train(train_data, print_freq)

            if scheduler is not None and self.epoch >0:
                scheduler.step()
            print("------ Testing epochs: {} ------".format(ep))
            self.test(test_data, print_freq)
            if (ep+1) % save_freq == 0:
                self.save(ep)

    def re_initialize(self):
        with torch.no_grad():
            for child in list(self.model.children()):
                print(child)
                for param in list(child.parameters()):
                    # if param.dim() == 2:
                    #   nn.init.xavier_uniform_(param)
                    nn.init.uniform_(param, a=-0.05, b=0.05)

    def save(self, epoch, loss=0, **kwargs):
        path = './relic_out/model/'
        if not os.path.exists(path):
            os.mkdir(path)
        for item in os.listdir(path):
            if item.startswith('%s_P%d' % (
                    self.network, self.percentage * 100)):
                open(path + item,
                     'w').close()  # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                os.remove(path + item)

        path_model = os.path.join(path, '%s_P%d_epoch%d' % (
            self.network, self.percentage * 100, epoch))
        save_checkpoint(self.model, epoch, self.optimizer, loss, path_model)

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
            for it, (in1, in2, len1, len2, label, semi, idx) in enumerate(data_loader):
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

if __name__ == '__main__':
    import torch.nn as nn
    lambda1 = lambda epoch: epoch // 30
    lambda2 = lambda epoch: 0.95 ** epoch
    model =  nn.Linear(10, 20)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
    for epoch in range(100):
        scheduler.step()
        print(optimizer.param_groups[0]['lr'])