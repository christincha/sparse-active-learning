from train.relicmulti_train import relic_multi_train
import torch
from ssTraining.kcenter_greedy import kCenterGreedy

class relic_multi_train_corset(relic_multi_train):
    def __init__(self, epoch, train_loader, eval_loader,
                 model, optimizer, cr_cla, cr_kl, k, writer,
                 network, device, T1, T2, af, labeled_bs, past_acc=0.3, percentage=0.05, en_num_l=3, hid_s=1024, few_knn=True, current_time= None):
        super().__init__(epoch, train_loader, eval_loader,
                 model, optimizer, cr_cla, cr_kl, k, writer,
                 network, device, T1, T2, af, labeled_bs, past_acc, percentage, en_num_l, hid_s, few_knn, current_time)

        self.features = torch.zeros(len(self.train_loader.dataset.label), 60).to(self.device)


    def concate_feature(self, seq_len =20):
        for it, (in1, in2, len1, len2, label, semi, idx) in enumerate(self.train_loader):
            in1 = in1.to(self.device)
            in2 = in2.to(self.device)
            pre1, pre2 = self.model.check_output(in1, in2, len1, len2)# num_heads * batch_size * 60
            self.features[idx, :] = pre1[0,:,:]


    def select_sam_index(self):
        self.concate_feature()
        if self.epoch == 0:
            self.cor_set = kCenterGreedy(self.features, self.train_loader.dataset.label, seed=1)
        else:
            self.cor_set.features = self.features
        if self.labeled_num < self.target_num:
            if self.labeled_num + 409 <= self.target_num:
                num_thisiter = 409
            else:
                num_thisiter = self.target_num - self.labeled_num
                num_thisiter = num_thisiter.cpu().numpy()
            #hi_train, label_train, index_train = remove_labeled_cluster(self.data_knn, self.label_knn, list(range(len(self.label_knn))), self.toLabel)
            # train_id_list, dis_list, dis_list_prob, cluster_label  = iter_kmeans_cluster(hi_train, label_train, index_train , ncluster=num_thisiter)
            # tmp = SampleNumber(train_id_list, dis_list, dis_list_prob, 'top', num_thisiter)
            #cor_set = kCenterGreedy(hi_train, label_train, seed=1)
            tmp = self.cor_set.select_batch_(None, self.toLabel, num_thisiter)
            self.toLabel = self.toLabel +tmp
            for i in range(len(tmp)):
                self.semi_label[tmp[i]] = self.train_loader.dataset.label[tmp[i]]
                self.select_ind.append(tmp[i])
                self.all_label.append(self.train_loader.dataset.data[tmp[i]])

            self.labeled_num += len(tmp)
            print('epoch : %d, finish one selection, selected %d sample' % (self.epoch, len(tmp)))


    def loop(self, epochs, train_data, test_data, scheduler=None, print_freq=-1, save_freq=1):

        for ep in range(epochs):
            self.epoch = ep
            if ep == 0:
                self.select_sam_index()
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
