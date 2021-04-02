# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from __future__ import unicode_literals, print_function, division
from ssTraining.SeqModel import seq2seq, SemiSeq2Seq
from model.Model import *
from train import *
#from ssTraining.seq_train import *
# load file

from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from data.data_loader import *
import torch
torch.cuda.set_device(0)

## training procedure
teacher_force = False
fix_weight = False
fix_state = True
few_knn = False
alpha = 0.5
max_length = 50
pro_tr = 0
pro_re = 0
phase  = 'RC'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = 'ssl_seq2seq'
# global variable
ProjectFolderName = 'NTUProject'
root_path = '/home/ws2/Documents/jingyuan/'
train_data = 'NTUtrain_cs_full.h5'
test_data = 'NTUtest_cs_full.h5'
# hyperparameter
feature_length = 75
hidden_size = 512
batch_size = 64
en_num_layers = 3
de_num_layers = 1
middle_size = 125
cla_num_layers = 1
learning_rate = 0.0001
epoch = 200
cla_dim = [60]

k = 2 # top k accuracy
# for classification
dataset_train = MySemiDataset(os.path.join(root_path, ProjectFolderName, train_data), 0.05, timediff=False)
dataset_test =  MySemiDataset(os.path.join(root_path, ProjectFolderName, test_data), 1,timediff=False)
# np.save('./labels/base_semiLabel.npy', dataset_train.semi_label)

semi_label = np.load('./labels/base_semiLabel.npy')
lb = np.copy(np.asarray(dataset_train.label))+1
lb[lb>60]=1
dataset_train.semi_label = lb#semi_label
for percentage in [0.05]:

    shuffle_dataset = True
    validation_split = 0.3
    dataset_size_train = len(dataset_train)
    dataset_size_test = len(dataset_test)

    # indices_unlabled = np.where(dataset_train.semi_label==0)[0]
    # indices_train = np.setdiff1d(np.arange(0, dataset_size_train), indices_unlabled).tolist()
    indices_train = list(range(dataset_size_train))
    indices_test = list(range(dataset_size_test))

    random_seed = 11111
    if shuffle_dataset :
      np.random.seed(random_seed)
      np.random.shuffle(indices_train)
      np.random.shuffle(indices_test)

    print("training data length: %d, validation data length: %d" % (len(indices_train), len(indices_test)))
    # seperate train and validation
    train_sampler = SubsetRandomSampler(indices_train)
    valid_sampler = SubsetRandomSampler(indices_test)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                               sampler=train_sampler, collate_fn=pad_collate_semi)
    eval_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                   sampler=valid_sampler, collate_fn=pad_collate_semi)


    # 0 non labeled class
    print_every = 1

    model = SemiSeq(feature_length, hidden_size, cla_dim).to(device)
    # model = SemiSeq2Seq(feature_length, hidden_size, feature_length, batch_size,
    #                          cla_dim, en_num_layers, de_num_layers, cla_num_layers, fix_state, fix_weight,
    #                          teacher_force)

    with torch.no_grad():
      for child in list(model.children()):
        print(child)
        for param in list(child.parameters()):
          # if param.dim() == 2:
          #   nn.init.xavier_uniform_(param)
            nn.init.uniform_(param, a=-0.05, b=0.05)
    #
    # model_name = './seq2seq_model/'+ 'selected_FSfewPCA0.0000_P100_layer3_hid1024_epoch30' #'FSfewCVrandomA0.0000_P50_layer3_hid1024_epoch16'  ##'FScvA0.0000_P100_layer3_hid1024_epoch4'#'FScvnewA0.0000_P100_layer3_hid1024_epoch1'#'test1_FWA0.0000_P100_layer3_hid1024_epoch255'
    # model_tmp =  seq2seq(feature_length, hidden_size, feature_length, batch_size,
    #                     en_num_layers, de_num_layers, fix_state, fix_weight, teacher_force)
    # optimizer_tmp = optim.Adam(filter(lambda p: p.requires_grad, model_tmp.parameters()), lr=learning_rate)
    # # #
    # model_tmp,_ = load_model(model_name, model_tmp, optimizer_tmp, device)
    # model.seq = model_tmp
    optimizer= optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    plot_pca = False
    data_pca = False
    predict = False
    loss_type = 'L1'  # 'L1'

    if loss_type == 'MSE':
        criterion_seq = nn.MSELoss(reduction='none')

    if loss_type == 'L1':
        criterion_seq = nn.L1Loss(reduction='none')

    criterion_cla = nn.CrossEntropyLoss(reduction='mean')

    past_acc = 0.4

    file_output = open('./output/Fourier_SSLP%d_en%d_hid%d_orL1.txt' % (
    percentage * 100, en_num_layers, hidden_size), 'w')
    print(' percentage %.2f' % (percentage))
    training(epoch, train_loader, eval_loader,
             model, optimizer,  criterion_cla, k, file_output, network=network, percentage=0.05, en_num_l=en_num_layers, hid_s=hidden_size)
    # training(epoch, train_loader, eval_loader, print_every,
    #          model, optimizer, criterion_seq, criterion_cla, alpha, k, file_output, past_acc,
    #          root_path, network, percentage, en_num_layers, hidden_size, num_class=10,
    #          few_knn=few_knn)