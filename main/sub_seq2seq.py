from __future__ import unicode_literals, print_function, division
from ssTraining.seq_train import *
from model.encoder_stack import Sub_Semi, seq2seq_stack
# load file
from label.label_statistic import average_label_prob
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from data.data_loader import *
import torch
from data.loader_construct import generate_dataloader
from utility.utilities import load_model
from train.subseq_train import *
from torch.utils.tensorboard import SummaryWriter
from utility.para_class import paramerters
torch.cuda.set_device(1)

class param_sub(paramerters):
    def __init__(self):
        super().__init__()
        self.teacher_force = False
        self.fix_weight = False
        self.fix_state = True
        self.few_knn = True
        self.percentage = 0.05
        self.alpha = 0.5
        self.max_length = 50
        self.pro_tr = 0
        self.pro_re = 0
        self.phase = 'RC'
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
        self.k = 2  # top k accuracy
        # for classificatio
        self.Checkpoint = False # load the whole model from check point
        self.pre_train = True
        self.old_modelName = '/home/ws2/Documents/jingyuan/sparse-active-learning-new/sparse-active-learning/OPC_granA0.0000_P100_layer3_hid1024_epoch30' #'/home/ws2/Documents/jingyuan/Self-Training/seq2seq_model/selected_FSfewPCA0.0000_P100_layer3_hid1024_epoch30'  # './seq2seq_model/' + 'test_seq2seq0_P5_epoch100' #'selected_FSfewPCA0.0000_P100_layer3_hid1024_epoch30'
        self.dataloader = MySemiDataset
        self.semi_label = []  # -1*np.ones(len(np.load('/home/ws2/Documents/jingyuan/Self-Training/labels/base_semiLabel.npy')))
        self.label_batch = 0
        self.print_every = 100
        self.save_freq = 10
        self.T1 = 0
        self.T2 = 1000
        self.af = 0.0001
        self.past_acc = 4

    def get_model(self):
        self.model = Sub_Semi(self.feature_length, self.hidden_size, self.feature_length, self.batch_size,
                                    self.cla_dim, self.en_num_layers, self.de_num_layers, self.cla_num_layers, self.fix_state, self.fix_weight,
                                    self.teacher_force)

        if self.pre_train and not self.Checkpoint:
            self.old_model = seq2seq_stack(self.feature_length, self.hidden_size, self.feature_length, self.batch_size,
                   self.en_num_layers, self.de_num_layers, self.fix_state, self.fix_weight,self.teacher_force).to(self.device)

if __name__ == '__main__':
    para = param_sub()
    for percentage in [para.percentage]:
    # model = SemiSeq2Seq(feature_length, hidden_size, feature_length, batch_size,
    #                          cla_dim, en_num_layers, de_num_layers, cla_num_layers, fix_state, fix_weight,
    #                          teacher_force)
        import socket
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(
            'subseq_out', current_time + '_' + socket.gethostname() + 'IC%d_en%d_hid%d_orL1.txt' % (
            percentage * 100, para.en_num_layers, para.hidden_size))

        with SummaryWriter(log_dir=log_dir) as writer:
            para.get_model()
            para.model_initialize()
            para.data_loader()
            para.get_optimizer()
            para.get_criterion()
            if para.Checkpoint or para.pre_train:
                para.load_model()
            para.scheduler()


            print(' percentage %.2f' % (para.percentage))
            # training(epoch, train_loader, eval_loader,
            #          model, optimizer,  criterion_cla, k, file_output, network='SSLbaseline', percentage=0.05, en_num_l=en_num_layers, hid_s=hidden_size)
            trainer = SubSeq_Train(para.epoch, para.train_loader, para.test_loader, para.print_every,
                     para.model, para.optimizer, para.criterion_seq, para.criterion_cla, para.alpha, para.k, writer, para.past_acc,
                     para.root_path, para.network, para.percentage, para.en_num_layers, para.hidden_size, para.label_batch,
                     few_knn=para.few_knn, TrainPS=para.Checkpoint, T1= para.T1, T2 = para.T2, af = para.af, current_time=current_time)
            trainer.loop(para.epoch,  para.train_loader, para.test_loader,
                         scheduler=para.model_scheduler, print_freq=para.print_every,
                         save_freq=para.save_freq)