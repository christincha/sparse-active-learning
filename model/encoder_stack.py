# defince NTU sub-granulitty

from ssTraining.EnDeModel import *
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

        return




if __name__ == '__main__':
    enstack = EncoderStack(3,2,2)
    intput = torch.randn([5,3, 75])
    out = enstack(intput, [3,3,3,3,3])