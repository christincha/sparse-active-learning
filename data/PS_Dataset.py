from data.augmentation import get_data_list
from torch.utils.data import Dataset
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence


class PSDataset(Dataset):
    def __init__(self, data_path, percentage, fourier=False, timediff = False, add_noise = False):

        self.data, self.label = get_data_list(data_path)
        # self.xy = zip(self.data, self.label)
        self.timediff = timediff
        self.add_noise = add_noise
        label = np.asarray(self.label)
        train_index = np.zeros(len(self.label))

        if percentage != 0:
            for i in range(1, 61):
                cur_ind = []
                for idx in range(len(label)):
                    if label[idx] == i:
                        cur_ind.append(idx)
                if len(cur_ind) != 0:
                    random.shuffle(cur_ind)
                    length = np.int(np.ceil((percentage * len(cur_ind))))
                    cur_ind = np.r_[cur_ind[:length]]
                    train_index[cur_ind] = i
        self.semi_label = train_index
        self.semi_old = np.copy(self.semi_label)
        self.label = np.asarray(label)


    def __getitem__(self, index):
        sequence = self.data[index]
        steps = sequence.shape[0]
        if not self.add_noise:
            if steps > 40:
                idx1 = np.random.choice(steps, 40)
                idx1 = np.sort(idx1)
                seq1 = sequence[idx1, :]
                idx2 = np.random.choice(steps, 40)
                idx2 = np.sort(idx2)
                seq2 = sequence[idx2, :]
            else:
                seq1, seq2 = sequence, sequence

        ##seq1 = rotation(seq1)
        seq2 = timediff_single(seq2)

        return seq1, seq2, self.label[index], self.semi_label[index], index

    def __len__(self):
        return len(self.label)

def timediff_single(data):
    return data[1:,:] - data[:-1,:]

def pad_collate_ps(batch):

    semi_label = [x[2] for x in batch]
    semi_label = np.asarray(semi_label)
    aug1= [torch.tensor(x[0], dtype=torch.float) for x in batch]
    #aug2  = ffttransform(data)
    aug2 = [torch.tensor(x[1], dtype=torch.float) for x in batch]
    label = np.asarray([x[2] for x in batch])
    semi_label = [x[3] for x in batch]
    semi_label = np.asarray(semi_label)
    index = np.asarray([x[4] for x in batch])
    #tddata = dct_transform(data)
    # aug1, aug2 = add_noise(da
    aug1_pad = pad_sequence(aug1, batch_first=True, padding_value=0)
    aug2_pad = pad_sequence(aug2, batch_first=True, padding_value=0)
    aug1_lens = [len(x) for x in aug1_pad]
    aug2_lens = [len(x) for x in aug2_pad]
    return aug1_pad, aug2_pad, aug1_lens, aug2_lens, label, semi_label, index

