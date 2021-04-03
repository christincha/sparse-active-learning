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
        for num_dis in [5,4,3,2,1]:
            sub = torch.where(count==num_dis)[0]
            mi = id_dif[sub,-1] - id_dif[sub, -2]
            vr1 = torch.argsort(mi)
            for i in range(len(vr1)):
                if unlab_id[sub[vr1[i]]]:
                    return sub[vr1[i]]
