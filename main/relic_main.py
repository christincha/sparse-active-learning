from __future__ import unicode_literals, print_function, division
import torch.nn as nn
import numpy as np
from train.relic_train import *
from model.relic_model import relic
from torch.nn import KLDivLoss, NLLLoss
from torch import optim
from data.relic_Dataset import MySemiDataset, pad_collate_semi, RotationDataset
#from siamesa.augmentation import MySemiDataset, pad_collate_semi
cr_kl = KLDivLoss(reduction='batchmean', log_target=True)
cr_cla = NLLLoss(reduction='mean')
from data.relic_Dataset import TwoStreamBatchSampler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler
import torch
from torch.utils.tensorboard import SummaryWriter
torch.cuda.set_device(1)

## training procedure
class paramerters():
    def __init__(self):
        self.phase  = 'RC'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = 'ssl_drop_ps'
        # global variable
        self.ProjectFolderName = 'NTUProject'
        self.root_path = '/home/ws2/Documents/jingyuan/'
        self.train_data = 'NTUtrain_cs_full.h5'
        self.test_data = 'NTUtest_cs_full.h5'
        # hyperparameter
        self.feature_length = 75
        self.hidden_size = 512
        self.batch_size = 64
        self.label_batch = 0
        self.en_num_layers = 3
        self.de_num_layers = 1
        self.middle_size = 125
        self.cla_num_layers = 1
        self.learning_rate = 0.0001
        self.epoch = 200
        self.cla_dim = [60]
        self.k = 2
        self.trained_model=None
        self.semi_label = None#np.load('../labels/base_semiLabel.npy')-1
        self.data_set = MySemiDataset
        self.model = relic(self.feature_length, self.hidden_size, self.cla_dim).to(self.device)
        # self.model = relic(self.feature_length, self.hidden_size, self.cla_dim, device=self.device,
        #           layout = 'ntu-rgb+d',strategy= 'distance',edge_weighting=True).to(self.device)
        self.print_every = 100
        self.past_acc = 0.4
        self.per = [0.05]
        self.T0 = 0
        self.T1 = 20
        self.af = 0.5
        self.device=(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_ind = False# notion to load model
        self.train_ps = False
        self.save_freq = 20

    def model_initialize(self):
        with torch.no_grad():
            for child in list(self.model.children()):
                print(child)
                for param in list(child.parameters()):
                    # if param.dim() == 2:
                    #   nn.init.xavier_uniform_(param)
                    nn.init.uniform_(param, a=-0.05, b=0.05)

    def get_optimizer(self):
        #self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.en2.parameters()), lr=self.learning_rate)

        self.optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, self.model.en2.parameters()),
        momentum=0.9,
        lr=self.learning_rate, weight_decay=0.0001)

    def load_model(self, model, model_name=None):
        model_name = '/home/ws2/Documents/jingyuan/Self-Training/relic/model/testssl_seq2seq_P5_epoch50'#ssl_drop_P5_epoch60'
        model, _ = load_model(model_name, model, self.optimizer, self.device)
        self.model = model
        self.get_optimizer()

    def data_loader(self):
        train_path = os.path.join(self.root_path, self.ProjectFolderName, self.train_data)
        test_path =  os.path.join(self.root_path, self.ProjectFolderName, self.test_data)

        self.train_loader, self.test_loader = generate_dataloader(train_path,
                                                                test_path,
                                                                  self.semi_label,
                                                                  self.batch_size,
                                                                  self.label_batch)

    def get_criterion(self):
        self.cr_kl = KLDivLoss(reduction='batchmean', log_target=True)
        self.cr_cla = NLLLoss(reduction='none', ignore_index=NO_LABEL)

    def scheduler(self):
        lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 3)
        self.model_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

 # top k accuracy
# for classification

# np.save('./labels/base_semiLabel.npy', dataset_train.semi_label)
def generate_dataloader(train_path, test_path, semi_label, batch_size, label_batch):
    dataset_train =MySemiDataset(train_path, 1)
    dataset_test = MySemiDataset(test_path, 1)
    if not semi_label:
        semi_label = -1*np.ones(len(dataset_train))
    unlabeled_idxs = np.where(semi_label==-1)[0]
    labeled_idxs = np.setdiff1d(range(len(dataset_train)), unlabeled_idxs)
    dataset_train.semi_label = semi_label
    assert len(dataset_train) == len(labeled_idxs) + len(unlabeled_idxs)
    if label_batch < batch_size and label_batch!=0:
        assert len(unlabeled_idxs) > 0
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, batch_size, label_batch)
    elif label_batch ==0:
        np.random.shuffle(unlabeled_idxs)
        sampler = SubsetRandomSampler(unlabeled_idxs)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
    else:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_sampler=batch_sampler,
                                               pin_memory=True, collate_fn=pad_collate_semi)

    eval_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              drop_last=False, collate_fn=pad_collate_semi)
    print("training data length: %d, validation data length: %d" % (len(dataset_train), len(dataset_test)))

    return train_loader, eval_loader

if __name__ == '__main__':
    para = paramerters()
    for percentage in para.per:
        import socket
        from datetime import datetime

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(
            'relic_out', current_time + '_' + socket.gethostname()+ 'SSLP%d_en%d_hid%d_orL1.txt' % (
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
            trainer = relic_train_copy(para.epoch, para.train_loader, para.test_loader,
                     para.model, para.optimizer,  para.cr_cla, para.cr_kl, para.k, writer,
                     para.network, para.device, para.T0, para.T1, para.af, para.label_batch ,  para.past_acc, percentage= percentage)

            trainer.loop(para.epoch, para.train_loader, para.test_loader,
                         scheduler=para.model_scheduler, print_freq=para.print_every, save_freq=para.save_freq)
            print(' percentage %.2f' % (percentage))
            # training(para.epoch, para.train_loader, para.test_loader,
            #          para.model, para.optimizer,  para.cr_cla, para.cr_kl, para.k, writer,
            #          para.network, para.past_acc, percentage= percentage)

    # training(epoch, train_loader, eval_loader, print_every,
    #          model, optimizer, criterion_seq, criterion_cla, alpha, k, file_output, past_acc,
    #          root_path, network, percentage, en_num_layers, hidden_size, num_class=10,
    #          few_knn=few_knn)