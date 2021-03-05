# load file
from __future__ import unicode_literals, print_function, division
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import os
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn

from torch import optim
import torch.nn.functional as F

import h5py
import numpy as np
import math
from torch.utils.data import random_split
import torchvision


torch.cuda.set_device(0)
import sys
from ssTraining.clustering_classification import *
from ssTraining.SeqModel import *
from utilities import *
from data_loader import *
from ssTraining.seq_train import *

## training procedure
teacher_force = False
fix_weight = False
fix_state = True
few_knn = True

phase  = 'PC'
if fix_weight:
  if few_knn:
    network = 'FWfew' + phase
  else:
    network = 'FW' + phase

if fix_state:
  if few_knn:
    network = 'FSfew' + phase
  else:
    network = 'FS' + phase

if not fix_state and not fix_weight:
  if few_knn:
    network = 'Ofew' + phase
  else:
    network = 'O' + phase

# hyperparameter
feature_length = 75
hidden_size = 1024
batch_size = 64
en_num_layers = 3
de_num_layers = 1
middle_size = 125
cla_num_layers = 1
learning_rate = 0.0001
epoch = 30

k = 2 # top k accuracy
# for classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# global variable
ProjectFolderName = 'NTUProject'
root_path = '/home/ws2/Documents/jingyuan/'+ ProjectFolderName

for percentage in [1]:

# percentage of classification

    alpha = 0
    data_path_train = root_path + '/NTUtrain_cs.h5'
    dataset_train = MySemiDataset(data_path_train, percentage)

    data_path_test = root_path + '/NTUtest_cs.h5'
    dataset_test = MySemiDataset(data_path_test, percentage)
    shuffle_dataset = True
    validation_split = 0.3
    dataset_size_train = len(dataset_train)
    dataset_size_test = len(dataset_test)

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


    cla_dim = [60]  # 0 non labeled class
    # class_weight = torch.ones(11)
    # class_weight[0] = 0
    # class_weight = class_weight.to(device)

    # initialize the model

    print_every = 1

    model = seq2seq(feature_length, hidden_size, feature_length, batch_size,
                        en_num_layers, de_num_layers, fix_state, fix_weight, teacher_force)

    # model = SemiSeq2Seq(feature_length, hidden_size, feature_length, batch_size,
    #                         cla_dim, en_num_layers, de_num_layers, cla_num_layers, fix_state, fix_weight, teacher_force)

    # with torch.no_grad():
    #     for child in list(model.children()):
    #         print(child)
    #         for param in list(child.parameters()):
    #             if param.dim() == 2:
    #                 # nn.init.xavier_uniform_(param)
    #                 nn.init.uniform_(param, a=-0.05, b=0.05)

    model_name = './seq2seq_model/'+ 'selected_FSfewPCA0.0000_P100_layer3_hid1024_epoch30' #'FSfewCVrandomA0.0000_P50_layer3_hid1024_epoch16'  ##'FScvA0.0000_P100_layer3_hid1024_epoch4'#'FScvnewA0.0000_P100_layer3_hid1024_epoch1'#'test1_FWA0.0000_P100_layer3_hid1024_epoch255'
    optimizer_tmp = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # # #
    model,_ = load_model(model_name, model, optimizer_tmp, device)
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

    criterion_cla = nn.CrossEntropyLoss(reduction='sum')

    past_acc = 10

    file_output = open(root_path + '/%sICA%.2f_P%d_en%d_hid%d_orL1.txt' % (
    network, alpha, percentage * 100, en_num_layers, hidden_size), 'w')
    print('network tye %s, alpha %.2d, percentage %.2f' % (network, alpha, percentage))
    training(epoch, train_loader, eval_loader, print_every,
             model, optimizer, criterion_seq, criterion_cla, alpha, k, file_output, past_acc,
             root_path, network, percentage, en_num_layers, hidden_size, num_class=60,
             few_knn=few_knn)

