from model.siamesa_model import EncoderRNN, FCN
import torch.nn as nn


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

