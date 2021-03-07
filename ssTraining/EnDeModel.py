# load file
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
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

import numpy as np
import math
from torch.utils.data import random_split
import torchvision
import torch.nn.init as init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# network module only set encoder to be bidirection
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.num_layers = num_layers


    def forward(self, input_tensor, seq_len):
        encoder_hidden = torch.Tensor().to(device)
        for it in range(max(seq_len)):
            if it == 0:
                enout_tmp, hidden_tmp = self.gru(input_tensor[:, it:it + 1, :])
            else:
                enout_tmp, hidden_tmp = self.gru(input_tensor[:, it:it + 1, :], hidden_tmp)
            encoder_hidden = torch.cat((encoder_hidden, enout_tmp), 1)

        hidden = torch.empty((1, len(seq_len), encoder_hidden.shape[-1])).to(device)
        count = 0
        for ith_len in seq_len:
            hidden[0, count, :] = encoder_hidden[count, ith_len - 1, :]
            count += 1
        # if hidden:
        #   output, hidden = self.gru(input, hidden)
        # else:
        #   output, hidden = self.gru(input)

        # output = self.out(output)
        return hidden  # 1*batch*featurelenghtu


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(output_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)

        output = self.out(output)
        return output, hidden


class Classification(nn.Module):
    def __init__(self, indim, out_dim, num_layers):
        super(Classification, self).__init__()
        self.out_dim = out_dim
        self.layers = num_layers
        self.indim = indim

        nn_list = []
        for i in range(num_layers):
            nn_list.append(nn.Linear(indim, self.out_dim[i]).to(device))
            if i != num_layers - 1:
                # nn_list.append(nn.Dropout(0.3).to(device))
                nn_list.append(nn.ReLU().to(device))
            indim = out_dim[i]
        self.linear = nn.ModuleList(nn_list)
        # self.softmax = nn.LogSoftmax(dim=1)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, input):

        for i, l in enumerate(self.linear):
            inter = input
            y = l(input)
            input = y

        out = y
        if self.layers == 1:
            assert inter.size()[-1] == self.indim
        inter = inter[np.newaxis, :]
        return out, inter



def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)




