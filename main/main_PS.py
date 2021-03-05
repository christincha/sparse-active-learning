from __future__ import unicode_literals, print_function, division
from label_generation import iter_label
from Model import *
from train import *
from ssTraining.SeqModel import SemiSeq2Seq
# load file
from label_statistic import summary_label_prob, average_label_prob, variance_ratio_label, average_label_base
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from data_loader import *
import torch
torch.cuda.set_device(1)
threshold = 0.9
drop = 0.
# parameters
ProjectFolderName = 'NTUProject'
root_path = '/home/ws2/Documents/jingyuan/'
train_data = 'NTUtrain_cs_full.h5'
test_data = 'NTUtest_cs_full.h5'
# hyperparameter
labeled_pr = 0.05
feature_length = 75
hidden_size = 1024
batch_size = 64
en_num_layers = 3
de_num_layers = 1
middle_size = 125
cla_num_layers = 1
learning_rate = 0.0000001
epoch = 300
cla_dim = [60]
k = 2
# 0 non labeled class
print_every = 1

model = SemiSeq(feature_length, hidden_size, cla_dim).to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
model_name = './seq2seq_model/' + 'ssl_seq2seqA0.5000_P5_layer3_hid1024_epoch139'
model_trained = SemiSeq2Seq(feature_length, hidden_size, feature_length, batch_size,
                            cla_dim, en_num_layers, de_num_layers, cla_num_layers)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_trained.parameters()), lr=learning_rate)

model_trained, _ = load_model(model_name, model_trained, optimizer, device)
model.encode = model_trained.seq.encoder
model.classifier = model_trained.classifier
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

dataset_train = MySemiDataset(os.path.join(root_path, ProjectFolderName, train_data), 0.05)
dataset_test =  MySemiDataset(os.path.join(root_path, ProjectFolderName, test_data), 1)
prob = np.load('./labels/SSLBase_ps_prob_full_rp1_drop%.2f.npy' % (drop))
semi_name = './labels/base_semiLabel.npy'
dataset_train.semi_old = np.load(semi_name)
dataset_ps = MyInterTrainDataset(os.path.join(root_path, ProjectFolderName, train_data), [], fourier=False)
dataset_ps.semi_label = np.load(semi_name)

train_loader_full = torch.utils.data.DataLoader(dataset_ps, batch_size=batch_size,
                               shuffle=False, collate_fn=pad_collate_iter)

for epoch in [100]:

    ps_semilabel = iter_label(0.99, dataset_train.semi_old, model, train_loader_full)
    dataset_train.semi_label = ps_semilabel
    shuffle_dataset = True
    dataset_size_train = len(dataset_train)
    dataset_size_test = len(dataset_test)

    indices_unlabled = np.where(dataset_train.semi_label==0)[0]
    indices_train = np.setdiff1d(np.arange(0, dataset_size_train), indices_unlabled).tolist()
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



    #
    # with torch.no_grad():
    #   for child in list(model.children()):
    #     print(child)
    #     for param in list(child.parameters()):
    #       # if param.dim() == 2:
    #       #   nn.init.xavier_uniform_(param)
    #         nn.init.uniform_(param, a=-0.05, b=0.05)
    # model_path = './trained_model/SSLbaselineAP5_layer3_hid1024_epoch47'
    # optimizer= optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # #model, _ = load_model(model_path, model, optimizer, device)


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

    file_output = open('./output/SSL_PS_Pes%d_en%d_hid%d_T%.2f.txt' % (
        labeled_pr * 100, en_num_layers, hidden_size, threshold), 'w')
    print(' percentage of labeled %.2f' % (labeled_pr))
    training(epoch, train_loader, eval_loader,
             model, optimizer,  criterion_cla, k, file_output, network='SSL_PS_Pes', percentage=labeled_pr, en_num_l=en_num_layers, hid_s=hidden_size)

