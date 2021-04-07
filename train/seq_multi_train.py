from ssTraining.SeqModel import *
from ssTraining.hidden_sample import *
from ssTraining.clustering_classification import remove_labeled_cluster
from  ssTraining.extract_hidden import test_extract_hidden_iter
from utility.utilities import *
from torch import optim
import torch.nn.functional as F
from  data.data_loader import *
from data.data_loader import NO_LABEL
#from  ssTraining.clustering_classification import *
import time
from  ssTraining.extract_hidden import *
from train.train_re_sel import ic_train

class recon_multi_train(ic_train):
    def __init__(self, epoch, train_loader, eval_loader, print_every,
             model, optimizer, criterion_seq, criterion_cla, alpha, k, writer, past_acc,
             root_path, network, percentage, en_num_layers, hidden_size, labeled_bs,
             few_knn=True, TrainPS=False, T1=0, T2 = 30, af = 0.3, current_time = None):
        super().__init__(epoch, train_loader, eval_loader, print_every,
             model, optimizer, criterion_seq, criterion_cla, alpha, k, writer, past_acc,
             root_path, network, percentage, en_num_layers, hidden_size, labeled_bs,
             few_knn, TrainPS, T1, T2, af, current_time)
        self.concate_data()
        self.select_knn()
  # select with consistency
    def select_sample_id(self, input_tensor, seq_len, unlab_id):
        pred_pro = self.model.check_output(input_tensor, seq_len)
        pred = torch.argmax(pred_pro, dim=-1)
        mod_value = torch.mode(pred.T, keepdim=True, dim=-1)
        count = torch.sum(pred.T==torch.cat([mod_value.values]*self.model.num_head, dim=-1), dim=-1)
        pos = torch.logical_and(count>=2, count<=4) #N

        pred_pro = torch.mean(pred_pro, dim=0)
        id_dif  = torch.sort(pred_pro, dim=-1)[0]
        # id_dif = id_dif[:,-1] - id_dif[:,-2] # margin difference
        # vr1 = torch.argsort(id_dif)
        # vr1 = list(range(len(seq_len)))
        # random.shuffle(vr1)
        # print(len(seq_len))
        # for i in range(len(vr1)):
        #     if unlab_id[vr1[i]]:# and pos[vr1[i]]: # select samples that has minimum MI but also has discrpancy between 5 output
        #         return vr1[i]
  #       for num_dis in [5,4,3,2,1]:
  #           sub = torch.where(count==num_dis)[0]
  #           mi = id_dif[sub,-1] - id_dif[sub, -2]
  #           vr1 = torch.argsort(mi)
  #           for i in range(len(vr1)):
  #               if unlab_id[sub[vr1[i]]]:
  #                   return sub[vr1[i]]
# select with variance
    def select_sample_id(self, input_tensor, seq_len, unlab_id):
        pred_pro = self.model.check_output(input_tensor, seq_len)
        pred = torch.argmax(pred_pro, dim=-1)
        variance = torch.var(pred_pro, dim=0)
        mean_var = torch.sum(variance, dim=-1)

        vr1 = torch.argsort(mean_var)
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
                # check the classification output
                #self.get_class(train_data, 'train_ep%d'%ep)

    def concate_data(self, seq_len=20):

        feature_len = self.train_loader.dataset.data[0].shape[-1]
        label_list = self.train_loader.dataset.label
        data_list = self.train_loader.dataset.data
        data = torch.tensor(())
        for i in range(len(label_list)):
            if data_list[i].size()[0] == seq_len:
                tmp = torch.flatten(data_list[i])
                print(tmp.size())
                data = torch.cat((data, tmp)).unsqueeze(0)
                print(data.size())

            if data_list[i].size()[0] < seq_len:
                dif = seq_len - data_list[i].size()[0]
                tmp = torch.cat((data_list[i], torch.zeros((dif, feature_len))))
                tmp = torch.flatten(tmp).unsqueeze(0)
                data = torch.cat((data, tmp))

            if data_list[i].size()[0] > seq_len:
                tmp = data_list[i][:seq_len, :]
                tmp = torch.flatten(tmp).unsqueeze(0)
                data = torch.cat((data, tmp))
        label_list = np.asarray(label_list)
        self.data_knn = data.numpy()
        self.label_knn = label_list

    def select_knn(self):
        toLabel  = []#list(torch.where(self.semi_label!=0)[0].cpu().numpy())
        hi_train, label_train, index_train = remove_labeled_cluster(self.data_knn, self.label_knn, list(range(len(self.label_knn))), toLabel)
        train_id_list, dis_list, dis_list_prob, cluster_label  = iter_kmeans_cluster(hi_train, label_train, index_train , ncluster=2005)
        tmp = SampleFromCluster(train_id_list, dis_list, dis_list_prob, 'top', 0.05)
        for i in range(len(tmp)):
            self.semi_label[tmp[i]] = self.train_loader.dataset.label[tmp[i]]
            self.select_ind[i] = self[tmp[i]]

        self.labeled_num += len(tmp)
        del self.data_knn