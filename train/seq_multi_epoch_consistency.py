from ssTraining.SeqModel import *
from ssTraining.hidden_sample import *
from ssTraining.clustering_classification import remove_labeled_cluster_inorder
from  ssTraining.extract_hidden import test_extract_hidden_iter
from utility.utilities import *
from ssTraining.kcenter_greedy import kCenterGreedy
from torch import optim
import torch.nn.functional as F
from  data.data_loader import *
from data.data_loader import NO_LABEL
#from  ssTraining.clustering_classification import *
import time
from  ssTraining.extract_hidden import *
from train.train_re_sel import ic_train

class recon_multiCon_train(ic_train):
    def __init__(self, epoch, train_loader, eval_loader, print_every,
             model, optimizer, criterion_seq, criterion_cla, alpha, k, writer, past_acc,
             root_path, network, percentage, en_num_layers, hidden_size, labeled_bs,
             few_knn=True, TrainPS=False, T1=0, T2 = 30, af = 0.3, current_time = None, toLabel=None):
        super().__init__(epoch, train_loader, eval_loader, print_every,
             model, optimizer, criterion_seq, criterion_cla, alpha, k, writer, past_acc,
             root_path, network, percentage, en_num_layers, hidden_size, labeled_bs,
             few_knn, TrainPS, T1, T2, af, current_time)
        self.select_ind = []
        self.result = []
        if not toLabel:
            self.toLabel = []
            self.sel_num = [401] * 5
            self.sel_count = 0
        else:
            self.toLabel = np.load(toLabel).tolist()
            self.sel_num = [401]*5
            self.sel_count = 1
            self.load_old_label()

        self.per_lab = np.zeros((epoch,2))
        self.save_path = './seq_multi_out/model/consistency'
  # select with consistency
    def mi(self, pred_pro):

        pred = torch.argmax(pred_pro, dim=-1)
        mod_value = torch.mode(pred.T, keepdim=True, dim=-1)
#        count = torch.sum(pred.T==torch.cat([mod_value.values]*self.model.num_head, dim=-1), dim=-1)
        #pos = torch.logical_or(count<2, count>4) #N

        #pred_pro = torch.mean(pred_pro, dim=0)
        id_dif  = torch.sort(pred_pro, dim=-1)[0]
        id_dif = id_dif[:,:,-1] - id_dif[:,:,-2] # margin difference
        id_dif = torch.sum(id_dif, dim=0)
        # if self.type != 'mi_prob':
        #     id_dif[pos] = 1
        return id_dif

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

        if is_train:
            self.result.append(sum(accuracy1)/labeled_n)
            if len(self.result) >3:
                self.result.pop(0)
                if torch.std(torch.tensor(self.result)) < 0.01:
                    self.select_sam_index()
        else:
            return sum(accuracy1)/labeled_n
    def loop(self, epochs, train_data, test_data, type='top', scheduler=None, print_freq=-1, save_freq=1, save_dir=None, start_epoch = 0):
        self.type = type
        iter_list = range(start_epoch, epochs)
        for ep in iter_list:
            self.per_lab[ep, 1] = self.labeled_num
            self.epoch = ep
            if ep == 0 and self.sel_count==0:
                self.select_sam_index()
            print("------ Training epochs: {} ------".format(ep))
            self.train(train_data, print_freq)
            if scheduler is not None:
                scheduler.step()
            print("------ Testing epochs: {} ------".format(ep))
            res = self.test(test_data, print_freq)
            self.per_lab[ep, 0] = res
            if ep % save_freq == 0:
                self.save(ep, path=save_dir)
                # check the classification output
                #self.get_class(train_data, 'train_ep%d'%ep)
            np.save(os.path.join(save_dir, 'result'), self.per_lab)
    def save(self, epoch, loss=0, path=None, path_model=None, **kwargs):
        if not path:
            path = './relic_multi_out/model/'
        if not os.path.exists(path):
            os.mkdir(path)
        # for item in os.listdir(path):
        #     if item.startswith('%s_P%d' % (
        #             self.network, self.percentage * 100)):
        #         open(path + item,
        #              'w').close()  # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
        #         os.remove(path + item)
        if not path_model:
            path_model = os.path.join(path, '%s_P%d_epoch%d' % (
                self.network, self.percentage * 100, epoch))
        save_checkpoint(self.model, epoch, self.optimizer, loss, path_model)
        np.save(path_model, self.toLabel)
    # concate the hidden state
    def concate_data(self, feature_size=2048):
        # extract hidden state
        train_length = len(self.train_loader.dataset)
        self.label_hi = np.zeros(train_length)
        hidden_train_tmp = torch.empty((train_length, feature_size)).to(device)
        label_list_train = np.zeros(( train_length), dtype=int)

        for ith, (ith_data, seq_len, label, semi, index) in enumerate(self.train_loader):
            input_tensor = ith_data.to(device)
            # label_list_train = label_list_train + labe
            en_hi= self.model.en_forward(input_tensor, seq_len)
            self.label_hi[index] = np.asarray(label)
            hidden_train_tmp[index, :] = en_hi[0, :, :]
        self.hidd = hidden_train_tmp.cpu().numpy()
    # compute the classifier output and the hidden state
    def concate_feature(self, feature_size=2048):
        # extract hidden state
        train_length = len(self.train_loader.dataset)
        self.label_hi = np.zeros(train_length)
        hidden_train_tmp = torch.empty((train_length, feature_size)).to(device)
        features = torch.zeros(train_length).to(device)

        for ith, (ith_data, seq_len, label, semi, index) in enumerate(self.train_loader):
            input_tensor = ith_data.to(device)
            # label_list_train = label_list_train + labe
            en_hi, probs= self.model.check_output(input_tensor, seq_len)
            features[index] = self.mi(probs)
            self.label_hi[index] = np.asarray(label)
            hidden_train_tmp[index, :] = en_hi[0, :, :]
        self.hidd = hidden_train_tmp.cpu().numpy()
        self.features = features.cpu().numpy()

    def select_sam_index(self, type = 'top'):
        #self.concate_data()
        if self.sel_count < len(self.sel_num):

            num_thisiter = self.sel_num[self.sel_count]

            # selection with different method
            if self.type == 'top':
                self.concate_data()
                hi_train, label_train, index_train = remove_labeled_cluster_inorder(self.hidd, self.label_hi, list(range(len(self.label_hi))), self.toLabel)
                train_id_list, dis_list, dis_list_prob, cluster_label  = iter_kmeans_cluster(hi_train, label_train, index_train , ncluster=num_thisiter)
                tmp = SampleNumber(train_id_list, dis_list, dis_list_prob, 'top', num_thisiter)
            if self.type == 'random':
                index = list(range(len(self.train_loader.dataset)))
                unlabeled = np.setdiff1d(index, self.toLabel)
                np.random.shuffle(unlabeled)
                tmp = unlabeled[:num_thisiter].tolist()
            if self.type == 'mi':
                self.concate_feature()
                hi_train, fea, index_train = remove_labeled_cluster_inorder(self.hidd, self.features,
                                                                            list(range(len(self.label_hi))),
                                                                            self.toLabel)
                train_id_list, dis_list, feat_list  = iter_kmeans_cluster_feature(hi_train, fea, index_train , ncluster=num_thisiter)
                if self.sel_count == 0:
                    tmp = SampleNumber(train_id_list, dis_list, feat_list, 'top', num_thisiter)
                else:
                    tmp = SampleNumber(train_id_list, dis_list, feat_list, 'mi', num_thisiter)

            if self.type =='mi_nocluster':
                self.concate_feature()
                hi_train, fea, index_train = remove_labeled_cluster_inorder(self.hidd, self.features,
                                                                            list(range(len(self.label_hi))),
                                                                            self.toLabel)
                pos = np.argsort(fea)
                index_train = np.asarray(index_train)
                tmp = index_train[pos[:num_thisiter]].tolist()

            if self.type == 'mi_prob':
                tmp = self.prob_mi(num_thisiter)
            if self.type == 'core_set':
                tmp = self.core_set(num_thisiter)

            self.toLabel = tmp + self.toLabel
            for i in range(len(tmp)):
                self.semi_label[tmp[i]] = self.train_loader.dataset.label[tmp[i]]
                self.select_ind.append(tmp[i])

            self.labeled_num += len(tmp)
            self.sel_count += 1

    def load_old_label(self):
        for i in range(len(self.toLabel)):
            self.semi_label[self.toLabel[i]] = self.train_loader.dataset.label[self.toLabel[i]]
            self.select_ind.append(self.toLabel[i])
            self.all_label.append(self.train_loader.dataset.data[self.toLabel[i]])

        self.labeled_num += len(self.toLabel)

    def prob_mi(self, num_thisiter):

        self.concate_feature()
        hi_train, fea, index_train = remove_labeled_cluster_inorder(self.hidd, self.features,
                                                                    list(range(len(self.label_hi))),
                                                                    self.toLabel)
        train_id_list, dis_list, feat_list = iter_kmeans_cluster_feature(hi_train, fea, index_train,
                                                                         ncluster=num_thisiter)
        if self.sel_count == 0:
            tmp = SampleNumber(train_id_list, dis_list, feat_list, 'top', num_thisiter)
        else:
            tmp = SampleNumber(train_id_list, dis_list, feat_list, 'mi_prob', num_thisiter)
        return tmp

    def core_set(self, num_thisiter):
        self.concate_feature()
        if self.epoch == 0:
            self.cor_set = kCenterGreedy(self.hidd, self.train_loader.dataset.label, seed=1)
        else:
            self.cor_set.features = self.hidd

        tmp = self.cor_set.select_batch_(None, self.toLabel, num_thisiter)
        return tmp