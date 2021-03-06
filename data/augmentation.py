from torch.fft import rfft
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
import random
import h5py
from scipy.fftpack import dct
from numpy.fft import fft, rfft, irfft
from data.rotationMatrix import rotation
def get_data_list(data_path, indices = None):
    f = h5py.File(data_path, 'r')
    data_list = []
    label_list = []
    if not indices:
        for i in range(len(f['label'])):

            if np.shape(f[str(i)][:])[0] > 10:
                x = f[str(i)][:]
                # original matrix with probability
                y = f['label'][i]

                #x = torch.tensor(x, dtype=torch.float)

                data_list.append(x)
                label_list.append(y)
    else:
        for i in indices:
            if np.shape(f[str(i)][:])[0] > 10:
                x = f[str(i)][:]
                # original matrix with probability
                y = f['label'][i]

                #x = torch.tensor(x, dtype=torch.float)

                data_list.append(x)
                label_list.append(y)

    return data_list, label_list

def ffttransform(data):
    n = len(data)
    fftdata = []
    for i in range(n):
        fftdata.extend([torch.tensor(np.absolute(fft(data[i], axis=0)), dtype=torch.float)])
    return fftdata

def timediff(data):
    n = len(data)
    tddata = []
    for i in range(n):
        tddata.extend([torch.tensor(data[i][1:,:]-data[i][:-1,:], dtype=torch.float)])
    return tddata

def relection(data):
    n = len(data)
    redata = []
    u = np.array([[1], [0],[0]])
    Q = np.eye(3) - 2*u@u.T

    for i in range(n):
        tmp = np.reshape(data[i], (len(data[i]), 3, -1))
        tmp = np.moveaxis(tmp, 0, -1)
        tmp = np.einsum('ij,jkn->ikn', Q, tmp)
        redata.extend([np.moveaxis(tmp, -1,0).reshape(len(data[i]), -1)])
    return redata


def add_noise(data):
    n = len(data)
    tddata1 = []
    tddata2 = []
    for i in range(n):

        tmp = np.copy(data[i])
        tmp = tmp
        steps = tmp.shape[0]
        if steps > 40:
            a = np.random.normal(0, 0.1, (2, 40, tmp.shape[1]))

            idx = np.random.choice(steps, 40, replace=False)
            idx = np.sort(idx)
            tddata1.extend([torch.tensor(tmp[idx,:] + a[0,:,:], dtype=torch.float)])
            idx = np.random.choice(steps, 40)
            idx = np.sort(idx)
            tddata2.extend([torch.tensor(tmp[idx,:] + a[1,:,:], dtype=torch.float)])
        else:
            a = np.random.normal(0, 0.1, (2, tmp.shape[0], tmp.shape[1]))
            tddata1.extend([torch.tensor(tmp + a[0,:,:], dtype=torch.float)])
            tddata2.extend([torch.tensor(tmp + a[1,:,:], dtype=torch.float)])


    return tddata1, tddata2

def dct_transform(data):
    n = len(data)
    dctdata = []
    for i in range(n):
        dctdata.extend([torch.tensor(dct(data[i], axis=0, type=2, norm = 'ortho'), dtype=torch.float)])
    return dctdata

def pad_collate_semi(batch):


    data = [x[0] for x in batch]
    label = [x[1] for x in batch]
    semi_label = [x[2] for x in batch]
    semi_label = np.asarray(semi_label)
    aug1= [torch.tensor(x[0], dtype=torch.float) for x in batch]
    #aug2  = ffttransform(data)
    aug2 = timediff(data)
    #tddata = dct_transform(data)
    # aug1, aug2 = add_noise(da
    aug1_pad = pad_sequence(aug1, batch_first=True, padding_value=0)
    aug2_pad = pad_sequence(aug2, batch_first=True, padding_value=0)
    aug1_lens = [len(x) for x in aug1_pad]
    aug2_lens = [len(x) for x in aug2_pad]
    return aug1_pad, aug2_pad, aug1_lens, aug2_lens, label, semi_label

def pad_collate_re(batch):

    label = [x[2] for x in batch]
    semi_label = [x[3] for x in batch]
    semi_label = np.asarray(semi_label)
    aug1= [torch.tensor(x[0], dtype=torch.float) for x in batch]
    #aug2  = ffttransform(data)
    aug2 = [torch.tensor(x[1], dtype=torch.float) for x in batch]
    #tddata = dct_transform(data)
    # aug1, aug2 = add_noise(da
    aug1_pad = pad_sequence(aug1, batch_first=True, padding_value=0)
    aug2_pad = pad_sequence(aug2, batch_first=True, padding_value=0)
    aug1_lens = [len(x) for x in aug1_pad]
    aug2_lens = [len(x) for x in aug2_pad]
    return aug1_pad, aug2_pad, aug1_lens, aug2_lens, label, semi_label


class MySemiDataset(Dataset):
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

    def __getitem__(self, index):
        sequence = self.data[index]
        steps = sequence.shape[0]
        if not self.add_noise:
            if steps > 40:
                idx = np.random.choice(steps, 40)
                idx = np.sort(idx)
                sequence = sequence[idx, :]


        label = self.label[index]
        semi_label = self.semi_label[index]

        return sequence, label, semi_label

    def __len__(self):
        return len(self.label)

class reflectDataset(Dataset):
    def __init__(self, data_path, percentage, reflect = True):

        self.data, self.label = get_data_list(data_path)
        # self.xy = zip(self.data, self.label)
        self.reflect = reflect
        if self.reflect:
            self.redata = relection(self.data)
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

    def __getitem__(self, index):
        sequence = self.data[index]
        re_seq = self.redata[index]
        steps = sequence.shape[0]

        if steps > 40:
            idx = np.random.choice(steps, 40)
            idx = np.sort(idx)
            idx2 = np.sort(np.random.choice(steps, 40))
            sequence = sequence[idx, :]
            re_seq = re_seq[idx2, :]


        label = self.label[index]
        semi_label = self.semi_label[index]

        return sequence, re_seq, label, semi_label

    def __len__(self):
        return len(self.label)

class rotationDataset(Dataset):
    def __init__(self, data_path, percentage, rotation= True):

        self.data, self.label = get_data_list(data_path)
        # self.xy = zip(self.data, self.label)
        self.rotation = rotation

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

    def __getitem__(self, index):
        sequence = self.data[index]
        steps = sequence.shape[0]

        if steps > 40:
            idx = np.random.choice(steps, 40)
            idx = np.sort(idx)
            idx2 = np.sort(np.random.choice(steps, 40))
            ro_seq1 = rotation(sequence[idx, :])
            ro_seq2 = rotation(sequence[idx2, :])
        else:
            ro_seq1 = rotation(sequence)
            ro_seq2 = rotation(sequence)


        label = self.label[index]
        semi_label = self.semi_label[index]

        return ro_seq1, ro_seq2, label, semi_label

    def __len__(self):
        return len(self.label)