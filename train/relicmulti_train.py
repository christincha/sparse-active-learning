from train.relic_train import relic_train_copy, nan_to_num
import os
from utility.utilities import *
from torch import optim
from data.relic_Dataset import NO_LABEL

class relic_multi_train(relic_train_copy):
    def __init__(self, epoch, train_loader, eval_loader,
                 model, optimizer, cr_cla, cr_kl, k, writer,
                 network, device, T1, T2, af, labeled_bs, past_acc=0.3, percentage=0.05, en_num_l=3, hid_s=1024, few_knn=True, current_time= None):
        super().__init__(epoch, train_loader, eval_loader,
                 model, optimizer, cr_cla, cr_kl, k, writer,
                 network, device, T1, T2, af, labeled_bs, past_acc, percentage, en_num_l, hid_s, few_knn, current_time)


    def select_sample_id(self, input1, input2, seq_len1, seq_len2, unlab_id):
        # mi is minimized
        # var is maximized
        pre1, pre2 = self.model.check_output(input1, input2, seq_len1,seq_len2) # num_heads * batch_size * 60
        ave_pro = torch.mean(torch.cat((pre1, pre2)), dim=0)

        id_dif = torch.sort(ave_pro)[0]
        id_dif = id_dif[:, -1] - id_dif[:, -2]  # margin difference
        vr1 = torch.argsort(id_dif)
        for i in range(len(vr1)):
            if unlab_id[vr1[i]]:
            # select samples that has minimum MI but also has discrpancy between 5 output
                return vr1[i]

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

                if self.labeled_num < self.target_num and (self.epoch+1)%5==0:
                    pos = self.select_sample_id(in1, in2, len1, len2, indicator)
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
                loss_kl = loss_kl.item()

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
                loss_kl = loss_kl.item()

            labeled_n += labeled_bs

            loop_losscla.append(loss_cla/ len(data_loader))
            loop_losskl.append(loss_kl/ len(data_loader))
            acc1 = semi.eq(p1.max(1)[1]).sum().item()
            acc2 = semi.eq(p2.max(1)[1]).sum().item()
            accuracy1.append(acc1)
            accuracy2.append(acc2)
            if print_freq > 0 and (it % print_freq) == 0:
                if is_train:
                    print(
                    f"[{mode}]loss[{it:<3}]\t labeled loss: {labeled_loss.item():.3f}\t new loss: {new_loss:.3f}\t"
                    f" loss kl: {loss_kl:.3f}\t  Acc1: {nan_to_num(acc1 / labeled_bs):.3%}\t"
                    f"Acc2 : {nan_to_num(acc2 / labeled_bs):.3%} labeled_num:{self.labeled_num:.1f} ")
                else:
                    print(
                        f"[{mode}]loss[{it:<3}]\t cla loss: {labeled_loss.item():.3f}\t"
                        f" loss kl: {loss_kl:.3f}\t Acc1: {acc1 / labeled_bs:.3%}\t"
                        f"Acc2 : {acc1 / labeled_bs:.3%} ")
            if self.writer:
                self.writer.add_scalar('loss/'+mode + '_global_loss_cla', loss_cla.item(), self.global_step)
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
                        np.save(os.path.join('relic_multi_out/label', self.current_time), self.select_ind)
                        self.save_label = False

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