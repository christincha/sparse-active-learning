# defince NTU sub-granulitty

from ssTraining.EnDeModel import *
from ssTraining.SeqModel import seq2seq, SemiSeq2Seq
def ntu_part():
    return {'body':[1, 2, 21], 'head':[21,3,4],
                    'LA':[5,6,7,8], 'RA':[9,10,11,12], 'LL':[12,13,15,16],
                    'RL':[17,18,19,20], 'LH': [22, 23], 'RH':[24,25]}


class EncoderStack(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers):
        super(EncoderStack, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.body_part = ntu_part()
        sub_enlist = []
        self.joints = []
        for key in self.body_part.keys():
            joint = self.body_part[key]
            self.joints.append(joint)
            sub_enlist.append(EncoderRNN(input_dim*len(joint), hidden_size, num_layers))
        self.encoders = nn.ModuleList(sub_enlist)

    def forward(self, input, seq_len):
        N,T, F = input.size()
        input = torch.reshape(input, (N, T, -1, self.input_dim))
        en_outputs = []
        for i, encoder in enumerate(self.encoders):
            joint = self.joints[i]
            cur_in = input[:,:, np.asarray(joint)-1, :]
            cur_in = torch.reshape(cur_in, (N,T,-1))
            hi = encoder(cur_in, seq_len)
            en_outputs.append(hi)
        fin =  torch.cat(en_outputs, dim=-1)
        return fin

class seq2seq_stack(seq2seq):
    def __init__(self, en_input_size, en_hidden_size, output_size, batch_size,
                 en_num_layers=3, de_num_layers=1,
                 fix_state=False, fix_weight=False, teacher_force=False, ntu=True):
        super().__init__(en_input_size, en_hidden_size, output_size, batch_size,
                 en_num_layers, de_num_layers,
                 fix_state, fix_weight, teacher_force)
        if ntu:
            sub_en_hi = int(en_hidden_size/8)
            dim = 3
            self.encoder = EncoderStack(dim, sub_en_hi, en_num_layers).to(device)
        else:
            raise KeyError('not NTU dataset')

class sub_classification(nn.Module):
    def __init__(self, indim, out_dim, num_layers):
        super(sub_classification,self).__init__()
        self.sub_part = 8
        self.sub_in_dim = int(indim/self.sub_part)
        self.out_dim = out_dim
        self.num_layers = num_layers
        sub_list = []
        nn_list = []


        for i in range(self.sub_part):
            sub_list.append(nn.Linear(self.sub_in_dim, self.out_dim[-1]).to(device))

        for i in range(num_layers):
            nn_list.append(nn.Linear(indim, self.out_dim[i]).to(device))
            if i != num_layers - 1:
                # nn_list.append(nn.Dropout(0.3).to(device))
                nn_list.append(nn.ReLU().to(device))
            indim = out_dim[i]

        self.nn_list = nn.ModuleList(nn_list)
        self.sub_list = nn.ModuleList(sub_list)

    def forward(self, input):
        out = []
        or_in = input
        for i, l in enumerate(self.nn_list):
            y = l(input)
            input = y
        out.append(y)

        for i, l in enumerate(self.sub_list):
            tmp = l(or_in[:, i*self.sub_in_dim:(i+1)*self.sub_in_dim])
            out.append(tmp)

        out = torch.stack(out)
        return out

class Sub_Semi(SemiSeq2Seq):
    def __init__(self, en_input_size, en_hidden_size, output_size, batch_size, cla_dim,
                 en_num_layers=3, de_num_layers=1, cl_num_layers=1,
                 fix_state=False, fix_weight=False, teacher_force=False, ntu=True
                 ):
        super().__init__(en_input_size, en_hidden_size, output_size, batch_size, cla_dim,
                 en_num_layers, de_num_layers, cl_num_layers,
                 fix_state, fix_weight, teacher_force)
        self.seq = seq2seq_stack(en_input_size, en_hidden_size, output_size, batch_size,
                 en_num_layers, de_num_layers,
                 fix_state, fix_weight, teacher_force, ntu)
        self.classifier = sub_classification(en_hidden_size*2, cla_dim, cl_num_layers)

    def forward(self, input_tensor, seq_len):
        encoder_hidden, deout = self.seq(input_tensor, seq_len)
        pred = self.classifier(encoder_hidden[0, :, :])

        # return encoder_hidden, deout, pred
        return encoder_hidden, deout, pred


if __name__ == '__main__':
    # enstack = EncoderStack(3,2,2)
    int_put = torch.randn([5,3, 75]).to(device)
    # out = enstack(intput, [3,3,3,3,3])
    feature = 75
    hidden = 16
    batch = 5
    cla_dim = [6]
    sub_smi = Sub_Semi(feature, hidden, feature, batch, cla_dim).to(device)
    hi, de_out, pre = sub_smi(int_put, [3,3,3,3,3])