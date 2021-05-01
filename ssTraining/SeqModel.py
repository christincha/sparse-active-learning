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
from ssTraining.EnDeModel import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class seq2seq(nn.Module):
    def __init__(self, en_input_size, en_hidden_size, output_size, batch_size,
                 en_num_layers=3, de_num_layers=1,
                 fix_state=False, fix_weight=False, teacher_force=False):
        super(seq2seq, self).__init__()
        self.batch_size = batch_size
        self.en_num_layers = en_num_layers
        self.encoder = EncoderRNN(en_input_size, en_hidden_size, en_num_layers).to(device)
        self.decoder = DecoderRNN(output_size, en_hidden_size * 2, de_num_layers).to(device)
        self.fix_state = fix_state
        self.fix_weight = fix_weight
        self.device = device
        if self.fix_weight:
            with torch.no_grad():
                # decoder fix weight
                self.decoder.gru.requires_grad = False
                self.decoder.out.requires_grad = False

        self.en_input_size = en_input_size
        self.teacher_force = teacher_force

    def forward(self, input_tensor, seq_len):
        self.batch_size = len(seq_len)

        encoder_hidden = self.encoder(
            input_tensor, seq_len)

        # tmp = encoder_hidden.view(self.en_num_layers, 2, batch_size, encoder.hidden_size)
        # decoder_hidden = torch.cat((tmp[self.en_num_layers-1:self.en_num_layers,0,:,:],
        #                             tmp[encoder.num_layers-1:encoder.num_layers,1,:,:]), 2)

        decoder_output = torch.Tensor().to(self.device)
        # decoder_hidden = encoder_hidden  # torch.empty((1,len(seq_len), out_seq.shape[-1]))
        # decoder part
        if self.teacher_force:
            de_input = torch.zeros([self.batch_size, 1, self.en_input_size], dtype=torch.float).to(device)
            de_input = torch.cat((de_input, input_tensor[:, 1:, :]), dim=1)
        else:
            de_input = torch.zeros(input_tensor.size(), dtype=torch.float).to(device)

        if self.fix_state:
            # de_input = torch.zeros([self.batch_size, 1, self.en_input_size], dtype=torch.float).to(device)

            # for it in range(max(seq_len)):
            #     deout_tmp, _ = self.decoder(
            #         de_input[:,it:it+1], encoder_hidden)
            #     decoder_output = torch.cat((decoder_output, deout_tmp), dim=1)

            de_input = input_tensor[:, 0:1,
                       :]  # torch.zeros([self.batch_size, 1, self.en_input_size], dtype=torch.float).to(device)

            for it in range(max(seq_len)):
                deout_tmp, _ = self.decoder(
                    de_input, encoder_hidden)
                deout_tmp = deout_tmp + de_input
                de_input = deout_tmp
                decoder_output = torch.cat((decoder_output, deout_tmp), dim=1)
        else:
            hidden = encoder_hidden
            for it in range(max(seq_len)):
                deout_tmp, hidden = self.decoder(
                    de_input[:, it:it + 1, :], hidden)

                decoder_output = torch.cat((decoder_output, deout_tmp), dim=1)
            # else:
            #     de_input = torch.zeros([self.batch_size, 1, self.en_input_size], dtype=torch.float).to(device)
            #     deout_tmp, hidden = self.decoder(
            #         de_input, hidden)
            #     for it in range(max(seq_len)):
            #         deout_tmp, hidden = self.decoder(
            #             de_input, hidden)
            #         #de_input = deout_tmp
            #         decoder_output = torch.cat((decoder_output, deout_tmp), dim=1)
        return encoder_hidden, decoder_output

class SemiSeq2Seq(nn.Module):
    def __init__(self, en_input_size, en_hidden_size, output_size, batch_size, cla_dim,
                 en_num_layers=3, de_num_layers=1, cl_num_layers=1,
                 fix_state=False, fix_weight=False, teacher_force=False):
        super(SemiSeq2Seq, self).__init__()

        self.seq = seq2seq(en_input_size, en_hidden_size, output_size, batch_size,
                           en_num_layers=en_num_layers, de_num_layers=de_num_layers,
                           fix_state=fix_state, fix_weight=fix_weight, teacher_force=teacher_force)
        self.classifier = Classification(en_hidden_size * 2, cla_dim, cl_num_layers)

    def forward(self, input_tensor, seq_len):
        encoder_hidden, deout = self.seq(input_tensor, seq_len)
        pred, inter = self.classifier(encoder_hidden[0, :, :])

        # return encoder_hidden, deout, pred
        return inter, deout, pred

    def en_cla_forward(self, input_tensor, seq_len):
        encoder_hidden = self.seq.encoder(input_tensor, seq_len)
        pred, inter = self.classifier(encoder_hidden[0,:,:])
        return  pred


class MultiSemiSeq2Seq(nn.Module):
    # quesstion: use multi-head, multi-classifier, or one head multi-diemension
    def __init__(self, en_input_size, en_hidden_size, output_size, batch_size, cla_dim,
                 en_num_layers=3, de_num_layers=1, cl_num_layers=1, num_head = 5, head_out_dim = 1024,
                 fix_state=False, fix_weight=False, teacher_force=False):
        super(MultiSemiSeq2Seq, self).__init__()
        self.num_head = num_head
        self.seq = seq2seq(en_input_size, en_hidden_size, output_size, batch_size,
                           en_num_layers=en_num_layers, de_num_layers=de_num_layers,
                           fix_state=fix_state, fix_weight=fix_weight, teacher_force=teacher_force)
        heads = []
        for i in range(num_head):
            heads.append(nn.Linear(en_hidden_size*2, head_out_dim).to(device))

                # nn_list.append(nn.Dropout(0.3).to(device))
            #heads.append(nn.ReLU().to(device))
        self.heads= nn.ModuleList(heads)

        self.classifier = Classification(head_out_dim, cla_dim, cl_num_layers)
        self.fix_head()

    def forward(self, input_tensor, seq_len):
        encoder_hidden, deout = self.seq(input_tensor, seq_len)

        # choose a random head
        random_head = random.randint(0, self.num_head-1)
        cur_head = self.heads[random_head]
        classifier_in = cur_head(encoder_hidden[0,:,:])
        pred, inter = self.classifier(classifier_in)

        # return encoder_hidden, deout, pred
        return encoder_hidden, deout, pred

    def en_cla_forward(self, input_tensor, seq_len):
        encoder_hidden = self.seq.encoder(input_tensor, seq_len)

        random_head = random.randint(0, self.num_head - 1)
        cur_head = self.heads[random_head]
        classifier_in = cur_head(encoder_hidden[0, :, :])
        pred, inter = self.classifier(classifier_in)
        return  pred

    def en_forward(self, input_tensor, seq_len):
        with torch.no_grad():
            hi = self.seq.encoder(input_tensor, seq_len)
        return hi

    def check_output(self, input_tensor, seq_len):
        with torch.no_grad():
            encoder_hidden = self.seq.encoder(input_tensor, seq_len)
            predict = []
            for i in range(self.num_head):
                cur_head = self.heads[i]
                classifier_in = cur_head(encoder_hidden[0, :, :])
                pred, inter = self.classifier(classifier_in)
                pred = torch.softmax(pred, dim=-1)
                predict.append(pred)

        return encoder_hidden, torch.stack(predict)

    def fix_head(self):
        for child in self.heads.children():
            for param in child.parameters():
                param.requires_grad = False