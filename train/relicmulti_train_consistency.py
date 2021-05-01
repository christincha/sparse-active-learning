from train.relicmulti_train import relic_multi_train
import torch
from ssTraining.kcenter_greedy import kCenterGreedy
import numpy as np
import os
class relic_multi_train_mi(relic_multi_train):
    def __init__(self, epoch, train_loader, eval_loader,
                 model, optimizer, cr_cla, cr_kl, k, writer,
                 network, device, T1, T2, af, labeled_bs, past_acc=0.3, percentage=0.05, en_num_l=3, hid_s=1024, few_knn=True, current_time= None):
        super().__init__(epoch, train_loader, eval_loader,
                 model, optimizer, cr_cla, cr_kl, k, writer,
                 network, device, T1, T2, af, labeled_bs, past_acc, percentage, en_num_l, hid_s, few_knn, current_time)

        self.features = torch.zeros(len(self.train_loader.dataset.label)).to(self.device)
        self.save_path = './relic_multi_out/model/consistency'
        self.sel_count = 9
        # weighted sampling
        self.toLabel = np.load('/home/ws2/Documents/jingyuan/sparse-active-learning-new/sparse_active_learning_EPOCH/sparse-active-learning/main/relic_multi_out/model/consistency/ssl_drop_ps_P10_epoch232.npy')
        print('running consistency')
        self.load_old_label()



    def concate_feature(self, seq_len =20):
        for it, (in1, in2, len1, len2, label, semi, idx) in enumerate(self.train_loader):
            in1 = in1.to(self.device)
            in2 = in2.to(self.device)
            pre1, pre2 = self.model.check_output(in1, in2, len1, len2)# num_heads * batch_size * 60
            # MI
            pre1 = torch.sort(pre1, dim=-1)[0]
            self.features[idx] = pre1[0, :, -1] - pre1[0, :, -2]

            # engtropy
            self.features[idx] = torch.sum(pre1[0,:,:] * torch.log(pre1[0,:,:]), dim=-1)
            # variance
            tmp = torch.std(torch.cat((pre1, pre2), dim=0), dim=-1)
            self.features[idx] = torch.mean(tmp, dim=0)

            # sampling sample with weight
            # average MI with consistency weight
            # ave_pro = torch.mean(pre1, dim=0)
            # id_dif = torch.sort(ave_pro)[0]
            # id_dif = id_dif[:, -1] - id_dif[:, -2]
            # pred = torch.argmax(torch.cat((pre1, pre2), dim=0), dim=-1)
            # mod_value = torch.mode(pred.T, keepdim=True, dim=-1)
            # count = torch.sum(pred.T == torch.cat([mod_value.values] * 2*self.model.num_head, dim=-1), dim=-1)
            # self.features[idx] = id_dif


    def select_sam_index(self):
        self.concate_feature()
        # if self.labeled_num < self.target_num:
        #     if self.labeled_num + 60 <= self.target_num:
        #         num_thisiter = 60
        #     else:
        #         num_thisiter = self.target_num - self.labeled_num
        #         num_thisiter = num_thisiter.cpu().numpy()
        self.save(self.epoch, path=self.save_path)
        if self.sel_count < len(self.sel_num):

            num_thisiter = self.sel_num[self.sel_count]
            self.sel_count += 1
            #hi_train, label_train, index_train = remove_labeled_cluster(self.data_knn, self.label_knn, list(range(len(self.label_knn))), self.toLabel)
            # train_id_list, dis_list, dis_list_prob, cluster_label  = iter_kmeans_cluster(hi_train, label_train, index_train , ncluster=num_thisiter)
            # tmp = SampleNumber(train_id_list, dis_list, dis_list_prob, 'top', num_thisiter)
            #cor_set = kCenterGreedy(hi_train, label_train, seed=1)
            #tmp = self.cor_set.select_batch_(None, self.toLabel, num_thisiter)
            index = list(range(len(self.features)))
            unlabeled = np.setdiff1d(index, self.toLabel)
            # choose the sample din't labeled and has with a lsmall
            if self.epoch == 0:
                np.random.shuffle(unlabeled)
                tmp = unlabeled[:num_thisiter].tolist()
            else:
                feature_id = torch.argsort(self.features[unlabeled], dim=0).cpu().numpy()            # transfer back to original id
                tmp  = unlabeled[feature_id[:num_thisiter]].tolist()
            self.toLabel = self.toLabel +tmp
            for i in range(len(tmp)):
                self.semi_label[tmp[i]] = self.train_loader.dataset.label[tmp[i]]
                self.select_ind.append(tmp[i])
                self.all_label.append(self.train_loader.dataset.data[tmp[i]])

            self.labeled_num += len(tmp)
            print('epoch : %d, finish one selection, selected %d sample' % (self.epoch, len(tmp)))


    def loop(self, epochs, train_data, test_data, scheduler=None, print_freq=-1, save_freq=1, save_dir = None):

        for ep in range(232, epochs):
            self.epoch = ep
            self.per_lab[ep, 1] = self.labeled_num
            if ep == 0:
                self.select_sam_index()
            print("------ Training epochs: {} ------".format(ep))
            # if (ep)%50==0:
            #     self.update_parameter()
            #     self.re_initialize()
            self.train(train_data, print_freq)

            if scheduler is not None and self.epoch >0:
                scheduler.step()
            print("------ Testing epochs: {} ------".format(ep))
            re = self.test(test_data, print_freq)
            self.per_lab[ep, 0 ] = re
            np.save(save_dir, self.per_lab)


