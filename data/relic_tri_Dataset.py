from torch.fft import rfft
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from data.relic_Dataset import MySemiDataset,TwoStreamBatchSampler, BatchSampler, SubsetRandomSampler
from data.PS_Dataset import timediff_single
from torch.utils.data.sampler import Sampler
import json
NO_LABEL = -1


class ThreestreamSemiDataset(MySemiDataset):
    def __init__(self, data_path,  percentage, bone_path = None, fourier=False, timediff = False, add_noise = False):
        super(ThreestreamSemiDataset, self).__init__(data_path, percentage, fourier, timediff, add_noise)
        if bone_path:
            with open(bone_path) as f:
                self.bone_data = json.load(f)['bonedata']
                self.bone_data = [np.asarray(x) for x in self.bone_data]

    def bone_transform(self, filename):
        self.bone = []
        bone_connect = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
                        (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22), \
                        (0, 16), (16, 17), (17, 18), (18, 19), (0, 12), (12, 13), (13, 14), (14, 15)]

        for x in self.data:
            T = x.shape[0]
            x = x.reshape((T, 25, 3))
            bone_x = np.zeros((T, 24,3))
            for i in range(len(bone_connect)):
                bone_x[:, i, :] = x[:, bone_connect[i][1], :] - x[:, bone_connect[i][0], :]
            bone_x = bone_x.reshape((T,-1))
            self.bone.append(bone_x.tolist())
        del self.data
        with open(filename, 'w') as f:
            json.dump({'bonedata': self.bone}, f)

    def __getitem__(self, index):
        sequence = self.data[index]
        sequence3 = self.data[index]
        steps = sequence.shape[0]
        if not self.add_noise:
            if steps > 40:
                idx1 = np.random.choice(steps, 40)
                idx1 = np.sort(idx1)
                sequence1 = sequence[idx1, :]
                idx2 = np.random.choice(steps, 40)
                idx2 = np.sort(idx2)
                sequence2 = sequence[idx2, :]
                idx3 = np.random.choice(steps, 40)
                idx3 = np.sort(idx3)
                sequence3 = sequence3[idx3, :]

            else:
                sequence1, sequence2 =  sequence, sequence

        sequence1, sequence2 = self.transform(sequence1, sequence2)

        label = self.label[index]
        semi_label = self.semi_label[index]
        return sequence1, sequence2, sequence3, label, semi_label, index

    def transform(self, sequence1, sequence2):
        return sequence1, timediff_single(sequence2)


def pad_collate_semi_tri(batch):

    label = [x[3] for x in batch]
    semi_label = [x[4] for x in batch]
    semi_label = np.asarray(semi_label)
    semi_old = [x[5] for x in batch]
    semi_old = np.asarray(semi_old)
    aug1= [torch.tensor(x[0], dtype=torch.float) for x in batch]
    #aug2  = ffttransform(data)
    aug2 = [torch.tensor(x[1], dtype=torch.float) for x in batch]
    aug3 = [torch.tensor(x[2], dtype=torch.float) for x in batch]
    #tddata = dct_transform(data)
    # aug1, aug2 = add_noise(da
    aug1_pad = pad_sequence(aug1, batch_first=True, padding_value=0)
    aug2_pad = pad_sequence(aug2, batch_first=True, padding_value=0)
    aug3_pad = pad_sequence(aug3, batch_first=True, padding_value=0)
    aug1_lens = [len(x) for x in aug1_pad]
    aug2_lens = [len(x) for x in aug2_pad]
    return aug1_pad, aug2_pad, aug3_pad, aug1_lens, aug2_lens, label, semi_label, semi_old

def tri_generate_dataloader(train_path, test_path, semi_label, batch_size, label_batch, pos=False):
    dataset_train =MySemiDataset(train_path, 1)
    dataset_test = MySemiDataset(test_path, 1)
    if len(semi_label)==0:
        semi_label = -1*np.ones(len(dataset_train))
    # pos decide where the sample is labled
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
                                               pin_memory=True, collate_fn=pad_collate_semi_tri)


    eval_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              drop_last=False, collate_fn=pad_collate_semi_tri)
    print("training data length: %d, validation data length: %d" % (len(dataset_train), len(dataset_test)))

    return train_loader, eval_loader