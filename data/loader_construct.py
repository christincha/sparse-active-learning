from data.data_loader import *
from data.sampler_twostream import TwoStreamBatchSampler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler
def generate_dataloader(loader_type, train_path, test_path, semi_label, batch_size, label_batch):
    dataset_train =loader_type(train_path, 1)
    dataset_test = loader_type(test_path, 1)
    if len(semi_label)==0:
        semi_label = -1*np.ones(len(dataset_train))
        dataset_train.semi_label = -1 * np.ones(len(dataset_train.semi_label))
    unlabeled_idxs = np.where(semi_label==-1)[0]
    labeled_idxs = np.setdiff1d(range(len(dataset_train)), unlabeled_idxs)
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
