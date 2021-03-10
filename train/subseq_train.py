from ssTraining.SeqModel import *
from utility.utilities import *
from torch import optim
import torch.nn.functional as F
from  data.data_loader import *
from data.data_loader import NO_LABEL
from  ssTraining.clustering_classification import *
import time
from  ssTraining.extract_hidden import *
from train.train_re_sel import ic_train

class SubSeq_Train(ic_train):
    def __init__(self, epoch, train_loader, eval_loader, print_every,
             model, optimizer, criterion_seq, criterion_cla, alpha, k, writer, past_acc,
             root_path, network, percentage, en_num_layers, hidden_size, labeled_bs,
             few_knn=True, TrainPS=False, T1=0, T2 = 30, af = 0.3, current_time = None):
        super().__init__(epoch, train_loader, eval_loader, print_every,
             model, optimizer, criterion_seq, criterion_cla, alpha, k, writer, past_acc,
             root_path, network, percentage, en_num_layers, hidden_size, labeled_bs,
             few_knn, TrainPS, T1, T2, af, current_time)

    def sub_loss(self,  cla_pre, label):
        loss_list = 0
        # computing the classification loss of each subnetwork
        for i in range(cla_pre.shape[0]):
            if i == 0:
                loss_list = self.cr_cla(cla_pre[i,:,:], label)
            else:
                loss_list += self.cr_cla(cla_pre[i,:,:], label)
        return loss_list

    def _iteration_step(self, input_tensor, seq_len, label, model, optimizer, criterion_seq, criterion_cla, alpha):
            optimizer.zero_grad()

            en_hi, de_out, cla_pre = model(input_tensor, seq_len)

            cla_loss = self.sub_loss(cla_pre, label)
            mask = torch.zeros([len(seq_len), max(seq_len)]).to(device)
            for ith_batch in range(len(seq_len)):
                mask[ith_batch, 0:seq_len[ith_batch]] = 1
            mask = torch.sum(mask, 1)

            seq_loss = torch.sum(criterion_seq(de_out, input_tensor), 2)
            seq_loss = torch.mean(torch.sum(seq_loss, 1) / mask)

            return cla_loss, seq_loss, en_hi, de_out.detach(), cla_pre

    def select_sample_id(self, unlab_id, cla_pre, seq_len, output):
        with torch.no_grad():
            #cla_pre_trans = self.model.en_cla_forward(output,seq_len)
            pre_o = torch.softmax(cla_pre, dim=-1)
            indices = torch.sort(pre_o)[-1][:,:,-1] # 9, 64 /9 multihead the first use all the subnetwork for prediction, the other for subnetwork
            cohherent = indices[1:,] == indices[0,:]
            cohherent = torch.sum(cohherent, dim=0)
            # vr1 = torch.argsort(dif)
            # for i in range(len(vr1)):
            #     if unlab_id[vr1[i]]:
            #         return vr1[i]
            vr1 = torch.argsort(cohherent)
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
            pos = -1
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
                if self.labeled_num < self.target_num and self.epoch%5==0:
                    labeled_bs +=1
                    pos = self.select_sample_id(indicator, cla_pre, seq_len,de_out)
                    self.labeled_num += 1
                    self.semi_label[id[pos]] = label[pos]
                    new_cla_loss = self.sub_loss(cla_pre[:,pos:pos+1, :], label[pos:pos+1])
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
                        np.save(os.path.join('reconstruc_out/label', self.currrent_time), self.all_label)
                        self.save_label = False

        def save(self, epoch, loss=0, **kwargs):
            path = './subnet_out/model/'
            if not os.path.exists(path):
                os.mkdir(path)
            for item in os.listdir(path):
                if item.startswith(os.path.join(path, '%s_P%d' % (
                        self.network, self.percentage * 100))):
                    open(path + item,
                         'w').close()  # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                    os.remove(path + item)

            path_model = os.path.join(path, '%s_P%d_epoch%d' % (
                self.network, self.percentage * 100, epoch))
            save_checkpoint(self.model, epoch, self.optimizer, loss, path_model)
