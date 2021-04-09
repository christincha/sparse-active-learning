import os
from model.relic_model import  relic_three
from torch.nn import KLDivLoss, NLLLoss
from train.relic_tri_train import relic_triple_train
from torch import optim
from data.relic_tri_Dataset import ThreestreamSemiDataset, tri_generate_dataloader
#from siamesa.augmentation import MySemiDataset, pad_collate_semi
cr_kl = KLDivLoss(reduction='batchmean', log_target=True)
cr_cla = NLLLoss(reduction='mean')
from data.relic_Dataset import TwoStreamBatchSampler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler
import torch
from torch.utils.tensorboard import SummaryWriter
torch.cuda.set_device(0)


from main.relic_main import paramerters

class relic_tri_para(paramerters):
    def __init__(self):
        super().__init__()
        self.num_head = 5
        self.head_dim = 1024
        self.model = relic_three(self.feature_length, self.hidden_size, self.cla_dim,
                                 en_num_layers=self.en_num_layers, cl_num_layers=self.cla_num_layers,
                                dropout=0).to(self.device)
        # self.model = relic_three(self.feature_length, self.hidden_size, self.cla_dim,
        #          en_num_layers=self.en_num_layers, cl_num_layers=self.cla_num_layers, num_head = self.num_head,
        #                              head_out_dim=self.head_dim,dropout=0).to(self.device)
        self.bone_train = 'NTUtrain_cs_bone.json'
        self.bone_test = 'NTUtest_cs_bone.json'

        self.model_ind = False  # notion to load model
        self.train_ps = False
        self.save_freq = 20

    def data_loader(self):
        train_path = os.path.join(self.root_path, self.ProjectFolderName, self.train_data)
        test_path =  os.path.join(self.root_path, self.ProjectFolderName, self.test_data)
        btrain = os.path.join(self.root_path, self.ProjectFolderName, self.bone_train)
        btest  = os.path.join(self.root_path, self.ProjectFolderName, self.bone_test)

        self.train_loader, self.test_loader = tri_generate_dataloader(train_path,
                                                                test_path,
                                                                  self.semi_label,
                                                                  self.batch_size,
                                                                  self.label_batch, btrain, btest, self.pos)



if __name__ == '__main__':
    para = relic_tri_para()
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
            trainer = relic_triple_train(para.epoch, para.train_loader, para.test_loader,
                     para.model, para.optimizer,  para.cr_cla, para.cr_kl, para.k, writer,
                     para.network, para.device, para.T0, para.T1, para.af, para.label_batch ,  para.past_acc, percentage= percentage, current_time=current_time)

            trainer.loop(para.epoch, para.train_loader, para.test_loader,
                         scheduler=para.model_scheduler, print_freq=para.print_every, save_freq=para.save_freq)
            print(' percentage %.2f' % (percentage))