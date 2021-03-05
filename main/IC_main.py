# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from __future__ import unicode_literals, print_function, division
from ssTraining.seq_train import *
from ssTraining.SeqModel import SemiSeq2Seq
from Model import SemiSeq
# load file
from label_statistic import average_label_prob
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from data.data_loader import *
import torch
from data.loader_construct import generate_dataloader
from utility.utilities import load_model
from ssTraining.ps_seq_train import *
from torch.utils.tensorboard import SummaryWriter
torch.cuda.set_device(0)

## training procedure
class paramerters:
    def __init__(self):
        self.teacher_force = False
        self.fix_weight = False
        self.fix_state = True
        self.few_knn = True
        self.percentage = 0.05
        self.alpha = 0.5
        self.max_length = 50
        self.pro_tr = 0
        self.pro_re = 0
        self.phase  = 'RC'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = 'seq2seq0'
        # global variable
        self.ProjectFolderName = 'NTUProject'
        self.root_path = '/home/ws2/Documents/jingyuan/'
        self.train_data = 'NTUtrain_cs_full.h5'
        self.test_data = 'NTUtest_cs_full.h5'
        # hyperparameter
        self.feature_length = 75
        self.hidden_size = 1024
        self.batch_size = 64
        self.en_num_layers = 3
        self.de_num_layers = 1
        self.middle_size = 125
        self.cla_num_layers = 1
        self.learning_rate = 0.0001
        self.epoch = 200
        self.cla_dim = [60]
        self.threshold = 0.8
        self.k = 2 # top k accuracy
        # for classificatio
        self.Trainps = True
        self.pre_train = False
        self.old_modelName = './seq2seq_model/' + 'test_seq2seq0_P5_epoch100' #'selected_FSfewPCA0.0000_P100_layer3_hid1024_epoch30'
        self.dataloader = MySemiDataset
        self.semi_label = np.load('/home/ws2/Documents/jingyuan/Self-Training/labels/base_semiLabel.npy')-1
        self.label_batch = 5
        self.print_every = 100
        self.save_freq=10
        self.T1 = 0
        self.T2 = 1000
        self.af = 0.0001

        self.per = [0.4]

        self.past_acc = 4

    def get_model(self):
        self.model = SemiSeq2Seq(self.feature_length, self.hidden_size, self.feature_length, self.batch_size,
                                    self.cla_dim, self.en_num_layers, self.de_num_layers, self.cla_num_layers, self.fix_state, self.fix_weight,
                                    self.teacher_force)

        if self.pre_train and not self.Trainps:
            self.old_model = seq2seq(self.feature_length, self.hidden_size, self.feature_length, self.batch_size,
                   self.en_num_layers, self.de_num_layers, self.fix_state, self.fix_weight,self.teacher_force).to(self.device)

    def model_initialize(self):
        with torch.no_grad():
            for child in list(self.model.children()):
                print(child)
                for param in list(child.parameters()):
                    # if param.dim() == 2
                    #   nn.init.xavier_uniform_(param)
                    nn.init.uniform_(param, a=-0.05, b=0.05)

    def get_optimizer(self):
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)

    def load_model(self):
        if self.pre_train:
            optimizer_tmp =  optim.Adam(filter(lambda p: p.requires_grad, self.old_model.parameters()), lr=self.learning_rate)
            model, _ = load_model(self.old_modelName, self.old_model, optimizer_tmp, self.device)
            self.model.seq = model
        if self.Trainps:
            optimizer_tmp =  optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
            self.model, _ = load_model(self.old_modelName, self.model, optimizer_tmp, self.device)
        self.get_optimizer()

    def data_loader(self):
        train_path = os.path.join(self.root_path, self.ProjectFolderName, self.train_data)
        test_path = os.path.join(self.root_path, self.ProjectFolderName, self.test_data)

        self.train_loader, self.test_loader = generate_dataloader(self.dataloader,train_path,
                                                                  test_path,
                                                                  self.semi_label,
                                                                  self.batch_size,
                                                                  self.label_batch)

    def get_criterion(self, loss_type='L1'):
        if loss_type == 'MSE':
            self.criterion_seq = nn.MSELoss(reduction='none')

        if loss_type == 'L1':
            self.criterion_seq = nn.L1Loss(reduction='none')

        self.criterion_cla = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL)

    def scheduler(self):
        lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 2)
        self.model_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)


# dataset_ps = MyInterTrainDataset(os.path.join(root_path, ProjectFolderName, train_data), [], fourier=False)
# dataset_ps.semi_label = np.load(semi_name)
# dataset_ps.semi_old = np.load(semi_name)
#
# train_loader_full = torch.utils.data.DataLoader(dataset_ps, batch_size=batch_size,
#                                shuffle=False, collate_fn=pad_collate_iter)
#
#model_tmp = SemiSeq(feature_length, hidden_size, cla_dim).to(device)
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
#model_name = './seq2seq_model/' + 'trained_seq2seq0A0.8000_P5_layer3_hid1024_epoch85'#'''ssl_seq2seqA0.5000_P5_layer3_hid1024_epoch139'
# 'FSfewCVrandomA0.0000_P50_layer3_hid1024_epoch16'  ##'FScvA0.0000_P100_layer3_hid1024_epoch4'#'FScvnewA0.0000_P100_layer3_hid1024_epoch1'#'test1_FWA0.0000_P100_layer3_hid1024_epoch255'
# model_trained, _ = load_model(model_name, model_trained, optimizer, device)
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_trained.parameters()), lr=learning_rate)

import copy
if __name__ == '__main__':
    para = paramerters()
    for percentage in para.per:
    # model = SemiSeq2Seq(feature_length, hidden_size, feature_length, batch_size,
    #                          cla_dim, en_num_layers, de_num_layers, cla_num_layers, fix_state, fix_weight,
    #                          teacher_force)

        with SummaryWriter(comment='./output/Rotation_SSLP%d_en%d_hid%d_orL1.txt' % (
            percentage * 100, para.en_num_layers, para.hidden_size)) as writer:
            para.get_model()
            para.model_initialize()
            para.data_loader()
            para.get_optimizer()
            para.get_criterion()
            if para.Trainps or para.pre_train:
                para.load_model()
            para.scheduler()


            file_output = open('./output/meancla_SSLP%d_en%d_hid%d_orL1.txt' % (
                para.percentage * 100, para.en_num_layers, para.hidden_size), 'w')
            print(' percentage %.2f' % (para.percentage))
            # training(epoch, train_loader, eval_loader,
            #          model, optimizer,  criterion_cla, k, file_output, network='SSLbaseline', percentage=0.05, en_num_l=en_num_layers, hid_s=hidden_size)
            trainer = ic_train(para.epoch, para.train_loader, para.test_loader, para.print_every,
                     para.model, para.optimizer, para.criterion_seq, para.criterion_cla, para.alpha, para.k, writer, para.past_acc,
                     para.root_path, para.network, para.percentage, para.en_num_layers, para.hidden_size, para.label_batch,
                     few_knn=para.few_knn, TrainPS=para.Trainps, T1= para.T1, T2 = para.T2, af = para.af)
            trainer.loop(para.epoch,  para.train_loader, para.test_loader,
                         scheduler=para.model_scheduler, print_freq=para.print_every,
                         save_freq=para.save_freq)
    # model_tmp.encode = model_trained.seq.encoder
    # model_tmp.classifier = model_trained.classifier
    # ps_semilabel = iter_label(0.99, dataset_train.semi_old, model_tmp, train_loader_full)
    # dataset_train.semi_label = ps_semilabel

    #model = SemiSeq(feature_length, hidden_size, cla_dim).to(device)
    # model = SemiSeq2Seq(feature_length, hidden_size, feature_length, batch_size,
    #                          cla_dim, en_num_layers, de_num_layers, cla_num_layers, fix_state, fix_weight,
    #                          teacher_force)

    # with torch.no_grad():
    #   for child in list(model.children()):
    #     print(child)
    #     for param in list(child.parameters()):
    #       # if param.dim() == 2:
    #       #   nn.init.xavier_uniform_(param)
    #         nn.init.uniform_(param, a=-0.05, b=0.05)
    #

