from model.siamesa_model import EncoderRNN, FCN
import torch.nn as nn
import torch
import random


class relic(nn.Module):
    def __init__(self, en_input_size, en_hidden_size, cla_dim,
                 en_num_layers=3, cl_num_layers=1, dropout=0
                 ):
        super(relic, self).__init__()

        self.en1= EncoderRNN(en_input_size, en_hidden_size, en_num_layers, dropout)
        self.en2 = EncoderRNN(en_input_size, en_hidden_size, en_num_layers, dropout)
        self.fcl1 = FCN(en_hidden_size * 2, cla_dim, len(cla_dim))
        self.fcl2 = FCN(en_hidden_size * 2, cla_dim, len(cla_dim))
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, input1, input2,  seq_len1,seq_len2):
        encoder_hidden1 = self.en1(input1, seq_len1)
        encoder_hidden2 = self.en2(input2, seq_len2)
        p1 = self.logsoftmax(self.fcl1(encoder_hidden1[0,:, :]))
        p2 = self.logsoftmax(self.fcl2(encoder_hidden2[0, :, :]))

        return encoder_hidden1, p1, encoder_hidden2, p2

class relic_multihead(nn.Module):
    def __init__(self, en_input_size, en_hidden_size, cla_dim,
                 en_num_layers=3, cl_num_layers=1, num_head = 5, head_out_dim=1024,dropout=0
                 ):
        super(relic_multihead, self).__init__()
        self.num_head = num_head
        self.en1= EncoderRNN(en_input_size, en_hidden_size, en_num_layers, dropout)
        self.en2 = EncoderRNN(en_input_size, en_hidden_size, en_num_layers, dropout)

        heads1 = []
        for i in range(num_head):
            heads1.append(nn.Linear(en_hidden_size*2, head_out_dim))

        heads2 = []
        for i in range(num_head):
            heads2.append(nn.Linear(en_hidden_size * 2, head_out_dim))

        self.heads1= nn.ModuleList(heads1)
        self.heads2 = nn.ModuleList(heads2)

        self.fcl1 = FCN(head_out_dim, cla_dim, len(cla_dim))
        self.fcl2 = FCN(head_out_dim, cla_dim, len(cla_dim))
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.fix_head()

    def forward(self, input1, input2,  seq_len1,seq_len2):
        encoder_hidden1 = self.en1(input1, seq_len1)
        encoder_hidden2 = self.en2(input2, seq_len2)

        # choose random head for head 1 and head 2
        r1 = random.randint(0, self.num_head-1)
        r2 = random.randint(0, self.num_head-1)

        head1 = self.heads1[r1](encoder_hidden1[0,:,:])
        head2 = self.heads2[r2](encoder_hidden2[0,:,:])

        p1 = self.logsoftmax(self.fcl1(head1))
        p2 = self.logsoftmax(self.fcl2(head2))


        return encoder_hidden1, p1, encoder_hidden2, p2

    def check_output(self, input1, input2, seq_len1, seq_len2):
        with torch.no_grad():
            predict1 = []
            predict2 = []

            encoder_hidden1 = self.en1(input1, seq_len1)
            encoder_hidden2 = self.en2(input2, seq_len2)
            for i in range(self.num_head):
                head1 = self.heads1[i](encoder_hidden1[0, :, :])
                head2 = self.heads2[i](encoder_hidden2[0, :, :])

                p1 = self.softmax(self.fcl1(head1))
                p2 = self.softmax(self.fcl2(head2))
                predict1.append(p1)
                predict2.append(p2)

        return torch.stack(predict1), torch.stack(predict2)

    def fix_head(self):
        for child in self.heads1.children():
            for param in child.parameters():
                param.requires_grad = False

        for child in self.heads2.children():
            for param in child.parameters():
                param.requires_grad = False


class relic_project(nn.Module):
    def __init__(self, en_input_size, en_hidden_size, cla_dim,
                 en_num_layers=3, cl_num_layers=1, num_head = 5, head_out_dim=1024,dropout=0
                 ):
        super(relic_project, self).__init__()
        self.num_head = num_head
        self.en1= EncoderRNN(en_input_size, en_hidden_size, en_num_layers, dropout)
        self.en2 = EncoderRNN(en_input_size, en_hidden_size, en_num_layers, dropout)

        heads1 = []
        for i in range(num_head):
            heads1.append(nn.Linear(en_hidden_size*2, head_out_dim))

        heads2 = []
        for i in range(num_head):
            heads2.append(nn.Linear(en_hidden_size * 2, head_out_dim))

        self.heads1= nn.ModuleList(heads1)
        self.heads2 = nn.ModuleList(heads2)

        self.fcl1 = FCN(head_out_dim, cla_dim, len(cla_dim))
        self.fcl2 = FCN(head_out_dim, head_out_dim, len(cla_dim))
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        #self.fix_head()

    def forward(self, input1, input2,  seq_len1,seq_len2):
        encoder_hidden1 = self.en1(input1, seq_len1)
        encoder_hidden2 = self.en2(input2, seq_len2)

        # choose random head for head 1 and head 2
        r1 = random.randint(0, self.num_head-1)
        r2 = random.randint(0, self.num_head-1)

        head1 = self.heads1[r1](encoder_hidden1[0,:,:])
        head2 = self.heads2[r2](encoder_hidden2[0,:,:])

        proj = self.fcl2(head2)
        p1 = self.logsoftmax(self.fcl1(head1))
        p2 = self.logsoftmax(self.fcl1(proj))


        return encoder_hidden1, p1, encoder_hidden2, p2

    def check_output(self, input1, input2, seq_len1, seq_len2):
        with torch.no_grad():
            predict1 = []
            predict2 = []

            encoder_hidden1 = self.en1(input1, seq_len1)
            encoder_hidden2 = self.en2(input2, seq_len2)
            for i in range(self.num_head):
                head1 = self.heads1[i](encoder_hidden1[0, :, :])
                head2 = self.heads2[i](encoder_hidden2[0, :, :])

                proj = self.fcl2(head2)
                p1 = self.softmax(self.fcl1(head1))
                p2 = self.softmax(self.fcl1(proj))
                predict1.append(p1)
                predict2.append(p2)

        return torch.stack(predict1), torch.stack(predict2)

    def fix_head(self):
        for child in self.heads1.children():
            for param in child.parameters():
                param.requires_grad = False

        for child in self.heads2.children():
            for param in child.parameters():
                param.requires_grad = False