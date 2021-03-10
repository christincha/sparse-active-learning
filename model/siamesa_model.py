import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch import optim
import torch.nn.functional as F

import numpy as np

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.num_layers = num_layers

    def forward(self, input_tensor, seq_len):
        # packed_input = pack_padded_sequence(input_tensor, seq_len, batch_first=True, enforce_sorted=False)
        # output, _ = self.gru(packed_input)
        # (out_seq, seq_len2) = pad_packed_sequence(output, batch_first=True)

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
        return hidden

class FCN(nn.Module):
    def __init__(self, indim=2048, out_dim=[512,2048], num_layers=2):
        super(FCN, self).__init__()
        self.out_dim = out_dim
        self.layers = num_layers
        self.indim = indim
        nn_list = []
        for i in range(num_layers):
            nn_list.append(nn.Linear(indim, self.out_dim[i]))
            #nn_list.append(nn.BatchNorm1d(self.out_dim[i]))
            if i!= num_layers-1:
                nn_list.append(nn.ReLU(inplace=True))
            indim = out_dim[i]
        self.linear = nn.ModuleList(nn_list)
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        for i, l in enumerate(self.linear):
            y = l(input)
            input = y
        return y


class SemiSeq(nn.Module):
    def __init__(self, en_input_size, en_hidden_size, cla_dim,
                 en_num_layers=3,  cl_num_layers=1, dropout= 0
                ):
        super(SemiSeq, self).__init__()

        self.encoder =  EncoderRNN(en_input_size, en_hidden_size, en_num_layers, dropout)
        self.predictor = FCN(en_hidden_size * 2, cla_dim, len(cla_dim))
        self.fcl = FCN(cla_dim[-1], cla_dim, len(cla_dim))

    def forward(self, input_tensor, seq_len):
        encoder_hidden = self.encoder(input_tensor, seq_len)
        encoder_hidden = self.predictor(encoder_hidden[0,:,:])
        p = self.fcl(encoder_hidden)

        return encoder_hidden , p