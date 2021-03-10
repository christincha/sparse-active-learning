from torch.fft import rfft
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
import random
import itertools
import h5py
from scipy.fftpack import dct
from numpy.fft import fft, rfft, irfft
from data.rotationMatrix import rotation, rotation_singe
from data.augmentation import get_data_list
from data.PS_Dataset import timediff_single
from torch.utils.data.sampler import Sampler
NO_LABEL = -1

class MySemiDataset(Dataset):
    def __init__(self, data_path, percentage, fourier=False, timediff = False, add_noise = False):

        self.data, self.label = get_data_list(data_path)
        self.label = [x-1 for x in self.label]
        # self.xy = zip(self.data, self.label)
        self.timediff = timediff
        self.add_noise = add_noise

        label = np.asarray(self.label)
        train_index = np.zeros(len(self.label))

        if percentage != 0 and percentage !=1:
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
        if percentage==1:
            train_index  = np.asarray(self.label)
        self.semi_label = train_index

    def __getitem__(self, index):
        sequence = self.data[index]
        steps = sequence.shape[0]
        if not self.add_noise:
            if steps > 40:
                idx1 = np.random.choice(steps, 40)
                idx1 = np.sort(idx1)
                sequence1 = sequence[idx1, :]
                idx2 = np.random.choice(steps, 40)
                idx2 = np.sort(idx2)
                sequence2 = sequence[idx2, :]

            else:
                sequence1, sequence2 = sequence, sequence

        sequence1, sequence2 = self.transform(sequence1, sequence2)

        label = self.label[index]
        semi_label = self.semi_label[index]
        return sequence1, sequence2, label, semi_label, index

    def transform(self, sequence1, sequence2):
        return sequence1, timediff_single(sequence2)

    def __len__(self):
        return len(self.label)

class RotationDataset(MySemiDataset):
    def __init__(self,  data_path, percentage, fourier=False, timediff = False, add_noise = False):
        MySemiDataset.__init__(self,  data_path, percentage, fourier, timediff, add_noise)

    def transform(self, sequence1, sequence2):
        return sequence1, rotation_singe(sequence2)

def pad_collate_semi(batch):

    label = [x[2] for x in batch]
    semi_label = [x[3] for x in batch]
    semi_label = np.asarray(semi_label)
    semi_old = [x[4] for x in batch]
    semi_old = np.asarray(semi_old)
    aug1= [torch.tensor(x[0], dtype=torch.float) for x in batch]
    #aug2  = ffttransform(data)
    aug2 = [torch.tensor(x[1], dtype=torch.float) for x in batch]
    #tddata = dct_transform(data)
    # aug1, aug2 = add_noise(da
    aug1_pad = pad_sequence(aug1, batch_first=True, padding_value=0)
    aug2_pad = pad_sequence(aug2, batch_first=True, padding_value=0)
    aug1_lens = [len(x) for x in aug1_pad]
    aug2_lens = [len(x) for x in aug2_pad]
    return aug1_pad, aug2_pad, aug1_lens, aug2_lens, label, semi_label, semi_old

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

from data.sampler_twostream import TwoStreamBatchSampler
from torch.utils.data import SubsetRandomSampler, BatchSampler
def generate_dataloader(train_path, test_path, semi_label, batch_size, label_batch, pos=False):
    dataset_train =MySemiDataset(train_path, 1)
    dataset_test = MySemiDataset(test_path, 1)
    if len(semi_label)==0:
        semi_label = -1*np.ones(len(dataset_train))
    if len(semi_label)!=0 and pos:
        tmp = -1*np.ones(len(dataset_train))
        for i in semi_label:
            i = int(i)
            tmp[i] = dataset_train.label[i]
        semi_label= tmp
    unlabeled_idxs = np.where(semi_label==-1)[0]
    labeled_idxs = np.setdiff1d(range(len(dataset_train)), unlabeled_idxs)
    dataset_train.semi_label = semi_label
    assert len(dataset_train) == len(labeled_idxs) + len(unlabeled_idxs)
    if label_batch < batch_size and label_batch!=0:
        assert len(unlabeled_idxs) > 0
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, batch_size, label_batch)
    elif label_batch ==0:
        np.random.shuffle(unlabeled_idxs)
        sampler = SubsetRandomSampler(unlabeled_idxs)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
    else:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_sampler=batch_sampler,
                                               pin_memory=True, collate_fn=pad_collate_semi)


    eval_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              drop_last=False, collate_fn=pad_collate_semi)
    print("training data length: %d, validation data length: %d" % (len(dataset_train), len(dataset_test)))

    return train_loader, eval_loader