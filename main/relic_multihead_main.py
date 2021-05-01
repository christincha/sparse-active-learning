import os
from model.relic_model import  relic_multihead
from train.relicmulti_train_consistency import relic_multi_train_mi
from torch.nn import KLDivLoss, NLLLoss
from utility.utilities import *
from torch import optim
from data.relic_Dataset import MySemiDataset, generate_dataloader
#from siamesa.augmentation import MySemiDataset, pad_collate_semi
cr_kl = KLDivLoss(reduction='batchmean', log_target=True)
cr_cla = NLLLoss(reduction='mean')
from data.relic_Dataset import TwoStreamBatchSampler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler
import torch
from torch.utils.tensorboard import SummaryWriter
torch.cuda.set_device(0)


from main.relic_main import paramerters

class relic_multi_para(paramerters):
    def __init__(self):
        super().__init__()
        self.num_head = 5
        self.head_dim = 1024
        self.epoch = 20000
        self.model = relic_multihead(self.feature_length, self.hidden_size, self.cla_dim,
                 en_num_layers=self.en_num_layers, cl_num_layers=self.cla_num_layers, num_head = self.num_head,
                                     head_out_dim=self.head_dim,dropout=0).to(self.device)

        self.model_ind = True  # notion to load model
        self.train_ps = False
        self.save_freq = 20

    def load_model(self, model, model_name=None):
        model_name = './relic_multi_out/model/consistency/ssl_drop_ps_P10_epoch232'#ssl_drop_P5_epoch60'
        model, _ = load_model(model_name, model, self.optimizer, self.device)
        self.model = model
        self.get_optimizer()



if __name__ == '__main__':
    para = relic_multi_para()
    for percentage in para.per:
        import socket
        from datetime import datetime

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(
            'relic_multi_out', current_time + '_' + socket.gethostname()+ 'SSLP%d_en%d_hid%d_orL1.txt' % (
            percentage * 100, para.en_num_layers, para.hidden_size))
        with SummaryWriter(log_dir=log_dir) as writer:

            para.model_initialize()
            para.data_loader()
            para.get_optimizer()
            para.get_criterion()
            if para.model_ind:
                para.load_model(para.model)
            para.scheduler()

        # file_output = open('./output/Rotation_SSLP%d_en%d_hid%d_orL1.txt' % (
        # percentage * 100, en_num_layers, hidden_size), 'w')
            trainer = relic_multi_train_mi(para.epoch, para.train_loader, para.test_loader,
                     para.model, para.optimizer,  para.cr_cla, para.cr_kl, para.k, writer,
                     para.network, para.device, para.T0, para.T1, para.af, para.label_batch ,  para.past_acc, percentage= percentage, current_time=current_time)
            res_dir = os.path.join(log_dir, 'result')
            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            trainer.loop(para.epoch, para.train_loader, para.test_loader,
                         scheduler=para.model_scheduler, print_freq=para.print_every, save_freq=para.save_freq, save_dir = res_dir)
            print(' percentage %.2f' % (percentage))