import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
import numpy as np
import torch
from ssTraining.SeqModel import MultiSemiSeq2Seq, seq2seq
from data.data_loader import MySemiDataset
from collections import Counter
from utility.para_class import paramerters
import seaborn as sns

def rand_cla_output(gt_label, random_label):
    pre_label = np.argmax(random_label, axis=-1)
    for i in range(1):
        cm = confusion_matrix(gt_label, pre_label[i,:])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        print(metrics.adjusted_rand_score(gt_label, pre_label[i,:]))
        print('Counter', Counter(gt_label[pre_label[i,:]==3]))
        disp.plot()
        plt.show()

        plt.figure(2)
        plt.hist([pre_label[i,:]], bins=np.arange(0.5,60.5,1))
        plt.show()

class ana_para(paramerters):
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
        self.Checkpoint = True
        self.pre_train = False
        self.old_modelName = '/home/ws2/Documents/jingyuan/sparse-active-learning-new/sparse-active-learning/main/reconstruc_out/model/resel_P5_epoch150' # './seq2seq_model/' + 'test_seq2seq0_P5_epoch100' #'selected_FSfewPCA0.0000_P100_layer3_hid1024_epoch30'
        self.dataloader = MySemiDataset
        self.semi_label = []  # -1*np.ones(len(np.load('/home/ws2/Documents/jingyuan/Self-Training/labels/base_semiLabel.npy')))
        self.label_batch = 0
        self.print_every = 100
        self.save_freq = 10
        self.T1 = 0
        self.T2 = 1000
        self.af = 0.0001

        self.past_acc = 4
        # parameters for head
        self.num_head = 5
        self.head_out_dim = 1024

    def get_model(self):
        self.model = MultiSemiSeq2Seq(self.feature_length, self.hidden_size, self.feature_length, self.batch_size,
                                      self.cla_dim, self.en_num_layers, self.de_num_layers, self.cla_num_layers,
                                      self.num_head, self.head_out_dim, self.fix_state, self.fix_weight,
                                      self.teacher_force)

        if self.pre_train and not self.Checkpoint:
            self.old_model = seq2seq(self.feature_length, self.hidden_size, self.feature_length, self.batch_size,
                                     self.en_num_layers, self.de_num_layers, self.fix_state, self.fix_weight,
                                     self.teacher_force).to(self.device)

class final_analysis():
    def __init__(self):
        para = ana_para()
        para.get_model()
        para.data_loader()
        para.load_model()
        self.para = para
        self.output = torch.zeros(self.para.model.num_head, len(self.para.train_loader.dataset), 60).to(self.para.device)

    def update_model(self):
        self.para.get_model()
        self.para.load_model()

    def extract(self):
        for it, (data, seq_len, label, semi, id) in enumerate(self.para.train_loader):
            input_tensor = data.to(self.para.device)
            self.output[:, id, :] = self.para.model.check_output(input_tensor, seq_len).detach()

    def analysis(self):
        import torch
        import pandas as  pd

        classes = torch.argmax(self.output, dim=-1)
        px = classes.cpu().numpy()
        px = pd.DataFrame(px.T)
        sns.pairplot(px)
        plt.show()

if __name__ == '__main__':
    gt_label = np.load('/home/ws2/Documents/jingyuan/sparse-active-learning-new/sparse-active-learning/main/reconstruc_out/result_ana/ground_truth.npy')
    random_label = np.load('/home/ws2/Documents/jingyuan/sparse-active-learning-new/sparse-active-learning/main/random_cla_output.npy')
    ana = final_analysis()
    for epoch in [0, 10, 50, 100]:
        model_name = 'resel_P5_epoch%d' % epoch
        ana.para.old_modelName = '/home/ws2/Documents/jingyuan/sparse-active-learning-new/sparse-active-learning/main/reconstruc_out/model/' +  model_name
        #ana.update_model()
        ana.extract()
        ana.analysis()