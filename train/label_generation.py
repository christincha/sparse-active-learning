from __future__ import unicode_literals, print_function, division

from Model import *
from train import *
from ssTraining.SeqModel import SemiSeq2Seq
# load file

from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from data_loader import *
import torch
torch.cuda.set_device(0)

def ps_labeling(data_loader, model, semi, rp = 1):
    #semi_label = torch.tensor(semi_label, dtype=torch.long).to(device)
    semi_label = torch.zeros(rp, len(semi), 60).to(device)
    softmax = nn.Softmax(dim=1)

    for iter_drop in range(rp):

        for it, (data, seq_len, label, _, index) in enumerate(data_loader):
            if it == 0:
                pass
            data = data.to(device)

            cla_pre_tmp = model(data, seq_len).detach()
            cla_pre_tmp = softmax(cla_pre_tmp)
            semi_label[iter_drop, index, :] = cla_pre_tmp
            # index = torch.tensor(index).to(device)
            # sub_set, ps_lab = torch.max(cla_pre, 1)
            # sub_set = sub_set > 0
            # ps_lab = ps_lab +1

    #np.save('./labels/Fourier_SSLBase_ps_prob_full_rp%d_drop%.2f.npy'% (rp, dropout), semi_label.cpu().numpy())
    return semi_label.cpu().numpy()

def average_label_base(threshold, semi, prob):
    semi_label = np.copy(semi)
    for drop in [0.3]:
        ave_prob = np.average(prob, axis=0)
        max_arg = np.argmax(ave_prob, axis=-1)
        max_prob = np.max(ave_prob, axis=-1)
        index = np.logical_and(semi_label==0,max_prob> threshold )
        semi_label[index] = max_arg[index]+1
    return semi_label, sum(index)

def iter_label(threshold,semi_label, model, data_loaader):
    prob = ps_labeling(data_loaader, model,semi_label)
    label,_ = average_label_base(threshold, semi_label, prob)
    # tr_label = np.asarray(data_loaader.dataset.label)
    # idex = np.logical_and(label!=0, label!=tr_label)
    # label[idex] = tr_label[idex]
    # label[label!=tr_label] = 0

    print('number of selected label %d correct %d ' %(sum(label!=0), sum(label[label!=0] ==np.asarray( data_loaader.dataset.label)[label!=0])))
    return label

if __name__ == '__main__':

    ## training procedure
    teacher_force = False
    fix_weight = True
    fix_state = False
    few_knn = False
    max_length = 50
    pro_tr = 0
    pro_re = 0
    phase  = 'RC'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # global variable
    ProjectFolderName = 'NTUProject'
    root_path = '/home/ws2/Documents/jingyuan/'
    train_data = 'NTUtrain_cs_full.h5'
    test_data = 'NTUtest_cs_full.h5'
    # hyperparameter
    feature_length = 75
    hidden_size = 1024
    batch_size = 64
    en_num_layers = 3
    de_num_layers = 1
    middle_size = 125
    cla_num_layers = 1
    learning_rate = 0.001
    epoch = 300
    cla_dim = [60]

    k = 2 # top k accuracy
    # for classification
    dataset_train = MyInterTrainDataset(os.path.join(root_path, ProjectFolderName, train_data), [], fourier=True)
    semi_label = np.load('./labels/base_semiLabel.npy')#np.zeros(len(dataset_train))
    indices_train = list(range(len(dataset_train)))#np.where(semi_label == 0)[0]
    train_sampler = SubsetRandomSampler(indices_train)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                               shuffle=False, collate_fn=pad_collate_iter)



    for drop_out in [0]:
        model = SemiSeq(feature_length, hidden_size, cla_dim, dropout=drop_out).to(device)
        optimizer= optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
       # model_name = './trained_model/' + #'SSLbaselineAP5_layer3_hid1024_epoch47'#'baselineAP5_layer3_hid1024_epoch193'
        model_name = './seq2seq_model/' + 'ssl_seq2seqA0.5000_P5_layer3_hid1024_epoch139'
        model_trained =  SemiSeq2Seq(feature_length, hidden_size, feature_length, batch_size,
                                 cla_dim, en_num_layers, de_num_layers, cla_num_layers, fix_state, fix_weight,
                                 teacher_force)
        optimizer= optim.Adam(filter(lambda p: p.requires_grad, model_trained.parameters()), lr=learning_rate)

        model_tmp, _ = load_model(model_name, model_trained, optimizer, device)
        model.encode = model_tmp.seq.encoder
        model.classifier = model_tmp.classifier
        model.train()
        threshold = 0.2
        ps_labeling(train_loader, model, semi_label, threshold)


