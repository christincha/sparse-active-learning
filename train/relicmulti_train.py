from train.relic_train import relic_train_copy, nan_to_num
from ssTraining.SeqModel import *
from ssTraining.hidden_sample import *
from ssTraining.clustering_classification import remove_labeled_cluster
import os
from utility.utilities import *
from torch import optim
from data.relic_Dataset import NO_LABEL
import queue
from ssTraining.kcenter_greedy import kCenterGreedy
class relic_multi_train(relic_train_copy):
    def __init__(self, epoch, train_loader, eval_loader,
                 model, optimizer, cr_cla, cr_kl, k, writer,
                 network, device, T1, T2, af, labeled_bs, past_acc=0.3, percentage=0.05, en_num_l=3, hid_s=1024, few_knn=True, current_time= None):
        super().__init__(epoch, train_loader, eval_loader,
                 model, optimizer, cr_cla, cr_kl, k, writer,
                 network, device, T1, T2, af, labeled_bs, past_acc, percentage, en_num_l, hid_s, few_knn, current_time)
        self.result = []
        self.toLabel = []
        self.concate_data(20)
        self.select_knn()


    def select_sample_id(self, input1, input2, seq_len1, seq_len2, unlab_id):
        # mi is minimized
        # var is maximized
        pre1, pre2 = self.model.check_output(input1, input2, seq_len1,seq_len2) # num_heads * batch_size * 60
        #ave_pro = torch.mean(torch.cat((pre1, pre2)), dim=0)
        pred = torch.argmax(pre1, dim=-1)
        mod_value = torch.mode(pred.T, keepdim=True, dim=-1)
        count = torch.sum(pred.T==torch.cat([mod_value.values]*self.model.num_head, dim=-1), dim=-1)
        pos = torch.logical_and(count>=1, count<=4) #N
        ave_pro = torch.mean(pre1, dim=0)
        id_dif = torch.sort(ave_pro)[0]
        id_dif = id_dif[:, -1] - id_dif[:, -2]  # margin difference
        vr1 = torch.argsort(id_dif)
        for i in range(len(vr1)):
            if unlab_id[vr1[i]] and pos[vr1[i]]:
            # select samples that has minimum MI but also has discrpancy between 5 output
                return vr1[i]
    # def select_sample_id(self, input1, input2, seq_len1, seq_len2, unlab_id):
    #     # mi is minimized
    #     # var is maximized
    #     pre1, pre2 = self.model.check_output(input1, input2, seq_len1,seq_len2) # num_heads * batch_size * 60
    #     var_pro = torch.var(torch.cat((pre1, pre2)), dim=0)
    #     ave_pro = torch.mean(var_pro, dim=-1)
    #     id_dif = torch.sort(ave_pro)[0]
    #     #id_dif = id_dif[:, -1] - id_dif[:, -2]  # margin difference
    #     vr1 = torch.argsort(id_dif)
    #     for i in range(len(vr1)):
    #         if unlab_id[vr1[i]]:
    #         # select samples that has minimum MI but also has discrpancy between 5 output
    #             return vr1[i]

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

                loss_kl = self.cr_kl(p1[indicator], p2[indicator]) / 2 + self.cr_kl(p2[indicator], p1[indicator]) / 2
                loss = loss_kl + labeled_loss
                loss_kl = loss_kl.item()
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
                loss_kl = loss_kl.item()

            labeled_n += labeled_bs

            loop_losscla.append(labeled_loss/ len(data_loader))
            loop_losskl.append(loss_kl/ len(data_loader))
            acc1 = semi.eq(p1.max(1)[1]).sum().item()
            acc2 = semi.eq(p2.max(1)[1]).sum().item()
            accuracy1.append(acc1)
            accuracy2.append(acc2)
            if print_freq > 0 and (it % print_freq) == 0:
                if is_train:
                    print(
                    f"[{mode}]loss[{it:<3}]\t labeled loss: {labeled_loss.item():.3f}\t  \t"
                    f" loss kl: {loss_kl:.3f}\t  Acc1: {nan_to_num(acc1 / labeled_bs):.3%}\t"
                    f"Acc2 : {nan_to_num(acc2 / labeled_bs):.3%} labeled_num:{self.labeled_num:.1f} ")
                else:
                    print(
                        f"[{mode}]loss[{it:<3}]\t cla loss: {labeled_loss.item():.3f}\t"
                        f" loss kl: {loss_kl:.3f}\t Acc1: {acc1 / labeled_bs:.3%}\t"
                        f"Acc2 : {acc2 / labeled_bs:.3%} ")
            if self.writer:
                self.writer.add_scalar('loss/'+mode + '_global_loss_cla', labeled_loss.item(), self.global_step)
                self.writer.add_scalar('loss/'+mode + '_global_loss_kl', loss_kl, self.global_step)
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
                    self.writer.add_histogram('hist/new_labeled', np.asarray(self.all_label), self.epoch, bins='sqrt')

        if is_train:
            self.result.append(sum(accuracy1)/labeled_n)
            if self.epoch >2:
                self.result.pop(0)
                if torch.std(torch.tensor(self.result)) < 0.01:
                    self.select_knn()

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

    def save(self, epoch, loss=0, **kwargs):
        path = './relic_multi_out/model/'
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


    def concate_data(self, seq_len=20):

        feature_len = self.train_loader.dataset.data[0].shape[-1]
        label_list = self.train_loader.dataset.label
        data_list = self.train_loader.dataset.data
        data_list = [torch.tensor(x) for x in data_list]
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
        if self.labeled_num < self.target_num:
            if self.labeled_num + 409 <= self.target_num:
                num_thisiter = 409
            else:
                num_thisiter = self.target_num - self.labeled_num
                num_thisiter = num_thisiter.cpu().numpy()
            hi_train, label_train, index_train = remove_labeled_cluster(self.data_knn, self.label_knn, list(range(len(self.label_knn))), self.toLabel)
            # train_id_list, dis_list, dis_list_prob, cluster_label  = iter_kmeans_cluster(hi_train, label_train, index_train , ncluster=num_thisiter)
            # tmp = SampleNumber(train_id_list, dis_list, dis_list_prob, 'top', num_thisiter)
            cor_set = kCenterGreedy(hi_train, label_train, seed=1)
            tmp = cor_set.select_batch_(None, self.toLabel, num_thisiter)
            self.toLabel.append(tmp)
            for i in range(len(tmp)):
                self.semi_label[tmp[i]] = self.train_loader.dataset.label[tmp[i]]
                self.select_ind.append(tmp[i])
                self.all_label.append(self.train_loader.dataset.data[tmp[i]])

            self.labeled_num += len(tmp)
            print('epoch : %d, finish one selection, selected %d sample' % (self.epoch, len(tmp)))

