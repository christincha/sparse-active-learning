## prepare data
# load file
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from label.sampling import *
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from label.label_statistic import *
import random
import torch
from label.sampling import *
import h5py
import numpy as np
from torch.fft import rfft
NO_LABEL = -1

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

                x = torch.tensor(x, dtype=torch.float)

                data_list.append(x)
                label_list.append(y)
    else:
        for i in indices:
            if np.shape(f[str(i)][:])[0] > 10:
                x = f[str(i)][:]
                # original matrix with probability
                y = f['label'][i]

                x = torch.tensor(x, dtype=torch.float)

                data_list.append(x)
                label_list.append(y)

    return data_list, label_list

def FourierTrasnform(data_list):
    n = len(data_list)
    for i in range(n):
        data_list[i] =  rfft(data_list[i],dim=0).abs()
    return data_list

def TimeDifference(data_list):
    n = len(data_list)
    for i in range(n):
        data_list[i] = data_list[i][1:,:] - data_list[i][:-1, :]
    return data_list

def concate_data(data_path, seq_len=10):
    data_list, label_list = get_data_list(data_path)

    feature_len = data_list[0].size()[-1]
    data = torch.tensor(())
    for i in range(len(label_list)):
        if data_list[i].size()[0] == seq_len:
            tmp = torch.flatten(data_list[i])
            data = torch.cat((data, tmp)).unsqueeze(0)

        if data_list[i].size()[0] < seq_len:
            dif = seq_len - data_list.size()[0]
            tmp = torch.cat((data_list[i], torch.zeros((dif, feature_len))))
            tmp = torch.flatten(tmp)
            data = torch.cat((data, tmp)).unsqueeze(0)

        if data_list[i].size()[0] > seq_len:
            tmp = data_list[i][:seq_len, :]
            tmp = torch.flatten(tmp).unsqueeze(0)
            data = torch.cat((data, tmp))
    label_list = np.asarray(label_list)
    return data.numpy(), label_lists


def pad_collate_semi(batch):
    lens = [len(x[0]) for x in batch]

    data = [x[0] for x in batch]
    label = [x[1] for x in batch]
    semi_label = [x[2] for x in batch]
    indice  = [x[3] for x in batch]
    semi_label = np.asarray(semi_label)
    indice = np.asarray(indice)
    xx_pad = pad_sequence(data, batch_first=True, padding_value=0)
    return xx_pad, lens, label, semi_label, indice


class MySemiDataset(Dataset):
    def __init__(self, data_path, percentage, fourier=False, timediff = False):

        self.data, self.label = get_data_list(data_path)
        self.label = [x-1 for x in self.label]
        # self.xy = zip(self.data, self.label)
        # self semi-label for semisupervised
        self.fourier = fourier
        self.timediff = timediff

        if fourier:
            self.data = FourierTrasnform(self.data)

        if timediff:
            self.data = TimeDifference(self.data)

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
        if not self.fourier:
            if steps > 40:
                idx = np.random.choice(steps, 40)
                idx = np.sort(idx)
                sequence = sequence[idx, :]
        else:
            if steps > 40:
                sequence = sequence[:40, :]
        label = self.label[index]
        semi_label = self.semi_label[index]

        return sequence, label, semi_label, index

    def __len__(self):
        return len(self.label)


class MyInterTrainDataset(Dataset):
    def __init__(self, data_path, label_position, ntu=False, fourier=False, timediff=False):

        self.data, self.label = get_data_list(data_path)
        # self.xy = zip(self.data, self.label)
        # self semi-label for semisupervised
        self.fourier = fourier
        self.timediff = timediff

        if fourier:
            self.data = FourierTrasnform(self.data)

        if timediff:
            self.data = TimeDifference(self.data)
        label = np.asarray(self.label)

        train_index = np.zeros(len(self.label))

        for idx in label_position:
            train_index[idx] = label[idx]

        self.semi_label = train_index
        self.index = list(range(len(label)))
        self.ntu = ntu

    def __getitem__(self, index):
        sequence = self.data[index]
        if self.ntu:
            step = sequence.shape[0]
            if step > 40:
                idx = np.random.choice(step, 40)
                idx = np.sort(idx)
                sequence = sequence[idx, :]
        label = self.label[index]
        semi_label = self.semi_label[index]
        ind = self.index[index]
        # Transform it to Tensor
        # x = torchvision.transforms.functional.to_tensor(sequence)
        # x = torch.tensor(sequence, dtype=torch.float)
        # y = torch.tensor([self.label[index]], dtype=torch.int)

        return sequence, label, semi_label, ind

    def __len__(self):
        return len(self.label)

class UnsupData(Dataset):
    def __init__(self, data_path, percentage=1):

        self.data, self.label = get_data_list(data_path)
        # self.xy = zip(self.data, self.label)
        # self semi-label for semisupervised

        label = np.asarray(self.label)
        train_index = np.ones(len(self.label))

        pos = list(range(len(self.label)))
        random.shuffle(pos)
        pos = np.r_[pos]

        self.semi_label = train_index

    def __getitem__(self, index):
        sequence = self.data[index]
        steps = sequence.shape[0]
        if steps > 40:
            idx = np.random.choice(steps, 40)
            idx = np.sort(idx)
            sequence = sequence[idx, :]
        label = self.label[index]
        semi_label = self.semi_label[index]

        return sequence, label, semi_label

    def __len__(self):
        return len(self.label)

def pad_collate_iter(batch):
    lens = [len(x[0]) for x in batch]
    data = [x[0] for x in batch]
    label = [x[1] for x in batch]
    semi_label = [x[2] for x in batch]
    semi_label = np.asarray(semi_label)
    idex = [x[3] for x in batch]
    idex = np.asarray(idex)
    xx_pad = pad_sequence(data, batch_first=True, padding_value=0)

    return xx_pad, lens, label, semi_label, idex


class UncertaintyData(Dataset):

    def __init__(self, data_path, prob_name, semi, percentage = 0.7, num_cla=60, sample = 'BD'):
        semi_base = np.load(semi)
        indices_unlabled = np.where(semi_base == 0)[0].tolist()
        indices_labeled = np.setdiff1d(np.arange(0, len(semi_base)), indices_unlabled).tolist()
        data_lab, label_lab = get_data_list(data_path,indices_labeled)
        data_unlab, label_unlab = get_data_list(data_path,indices_unlabled)
        y_mean, y_var, y_pred, y_T = mc_dropout_evaluate(prob_name, semi, classes=num_cla)
        # self.xy = zip(self.data, self.label)
        # self semi-label for semisupervised
        num_sample = int(len(semi_base)*percentage)
        if sample == 'BD':
            X_s, y_s, w_s = sample_by_bald_difficulty(data_unlab, y_mean, y_var, y_pred, num_sample, num_cla, y_T)

        if sample == 'BE':
             X_s, y_s, w_s =sample_by_bald_difficulty(data_unlab, y_mean, y_var, y_pred, num_sample, num_cla, y_T)

        if sample == 'BCE':
            X_s, y_s, w_s = sample_by_bald_class_easiness(data_unlab, y_mean, y_var, y_pred, num_sample, num_cla, y_T)

        if sample == 'BCD':
            X_s, y_s, w_s = sample_by_bald_class_difficulty(data_unlab, y_mean, y_var, y_pred, num_sample, num_cla, y_T)
        self.data = data_lab + X_s
        self.label = np.hstack((label_lab , y_s))
        self.ws = np.hstack((np.zeros(len(label_lab)), w_s))
        self.semi_label = np.hstack((np.ones(len(label_lab)), np.zeros(len(label_unlab))))

    def __getitem__(self, index):
        sequence = self.data[index]
        steps = sequence.shape[0]
        if steps > 40:
            idx = np.random.choice(steps, 40)
            idx = np.sort(idx)
            sequence = sequence[idx, :]
        label = self.label[index]
        semi_label = self.semi_label[index]
        ws = self.ws[index]
        return sequence, label, semi_label, ws

    def __len__(self):
        return len(self.label)






