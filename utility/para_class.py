from __future__ import unicode_literals, print_function, division
from ssTraining.seq_train import *
from ssTraining.SeqModel import SemiSeq2Seq
from model.Model import SemiSeq
# load file
from label.label_statistic import average_label_prob
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from data.data_loader import *
import torch
from data.loader_construct import generate_dataloader
from utility.utilities import load_model
from train.train_re_sel import *
from torch.utils.tensorboard import SummaryWriter
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
        self.network = 'resel'
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
        self.Checkpoint= False
        self.pre_train = True
        self.old_modelName = '/home/ws2/Documents/jingyuan/Self-Training/seq2seq_model/selected_FSfewPCA0.0000_P100_layer3_hid1024_epoch30'#'./seq2seq_model/' + 'test_seq2seq0_P5_epoch100' #'selected_FSfewPCA0.0000_P100_layer3_hid1024_epoch30'
        self.dataloader = MySemiDataset
        self.semi_label = []#-1*np.ones(len(np.load('/home/ws2/Documents/jingyuan/Self-Training/labels/base_semiLabel.npy')))
        self.label_batch = 0
        self.print_every = 100
        self.save_freq=10
        self.T1 = 0
        self.T2 = 1000
        self.af = 0.0001

        self.past_acc = 4

    def get_model(self):
        self.model = SemiSeq2Seq(self.feature_length, self.hidden_size, self.feature_length, self.batch_size,
                                    self.cla_dim, self.en_num_layers, self.de_num_layers, self.cla_num_layers, self.fix_state, self.fix_weight,
                                    self.teacher_force)

        if self.pre_train and not self.Checkpoint:
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
            self.get_optimizer()
        if self.Checkpoint:
            optimizer_tmp =  optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
            self.model, self.optimizer = load_model(self.old_modelName, self.model, optimizer_tmp, self.device)


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