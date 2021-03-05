import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test_extract_hidden_semi(model, data_train, data_eval, alpha):
    label_list_train = []
    label_list_eval = []
    feature_size = 2048
    repeat = 1
    train_length = len(data_train.dataset)
    eval_len = len(data_eval.dataset)
    hidden_train_tmp = torch.empty((repeat * train_length, feature_size)).to(device)
    hidden_eval_tmp = torch.empty((repeat * eval_len, feature_size)).to(device)
    label_train_semi = np.zeros(repeat * train_length, dtype=int)
    label_eval_semi = np.zeros(repeat * eval_len, dtype=int)
    label_train_iter = np.zeros(repeat * train_length, dtype=int)
    label_eval_iter = np.zeros(repeat * eval_len, dtype=int)

    label_list_train = np.zeros((repeat * train_length), dtype=int)
    label_list_eval = np.zeros((repeat * eval_len), dtype=int)

    hidden_array_train = np.zeros((repeat * train_length, feature_size))
    hidden_array_eval = np.zeros((repeat * eval_len, feature_size))

    for isample in range(repeat):
        start = train_length * isample
        for ith, (ith_data, seq_len, label, semi,_) in enumerate(data_train):
            input_tensor = ith_data.to(device)
            # label_list_train = label_list_train + label

            step = ith_data.shape[0]
            if alpha == 0:
                # print(input_tensor.size(), len(seq_len))
                en_hi, de_out = model(input_tensor, seq_len)
                cla_pre = None
            else:
                en_hi, de_out, cla_pre = model(input_tensor, seq_len)
            label_list_train[start:start + step] = np.asarray(label)
            label_train_semi[start:start + step] = semi
            hidden_train_tmp[start:start + step, :] = en_hi[0, :, :].detach()
            start = start + step

    for isample in range(repeat):
        start = isample * eval_len
        for ith, (ith_data, seq_len, label, semi,_) in enumerate(data_eval):
            step = ith_data.shape[0]
            input_tensor = ith_data.to(device)
            if alpha == 0:
                en_hi, de_out = model(input_tensor, seq_len)
                cla_pre = None
            else:
                en_hi, de_out, cla_pre = model(input_tensor, seq_len)
            label_list_eval[start:start + step] = np.asarray(label)
            label_eval_semi[start:start + step] = semi
            hidden_eval_tmp[start:start + step, :] = en_hi[0, :, :].detach()
            start = start + step

    return hidden_train_tmp.cpu().numpy(), hidden_eval_tmp.cpu().numpy(), label_list_train.tolist(), label_list_eval.tolist(), label_train_semi, label_eval_semi

def test_extract_hidden_ps(model, data_train, data_eval, alpha):
    label_list_train = []
    label_list_eval = []
    feature_size = 2048
    repeat = 1
    train_length = len(data_train.dataset)
    eval_len = len(data_eval.dataset)
    hidden_train_tmp = torch.empty((repeat * train_length, feature_size)).to(device)
    hidden_eval_tmp = torch.empty((repeat * eval_len, feature_size)).to(device)
    label_train_semi = np.zeros(repeat * train_length, dtype=int)
    label_eval_semi = np.zeros(repeat * eval_len, dtype=int)
    label_train_iter = np.zeros(repeat * train_length, dtype=int)
    label_eval_iter = np.zeros(repeat * eval_len, dtype=int)

    label_list_train = np.zeros((repeat * train_length), dtype=int)
    label_list_eval = np.zeros((repeat * eval_len), dtype=int)

    hidden_array_train = np.zeros((repeat * train_length, feature_size))
    hidden_array_eval = np.zeros((repeat * eval_len, feature_size))

    for isample in range(repeat):
        start = train_length * isample
        for ith, (ith_data, seq_len, label, semi,_) in enumerate(data_train):
            input_tensor = ith_data.to(device)
            # label_list_train = label_list_train + label

            step = ith_data.shape[0]
            if alpha == 0:
                # print(input_tensor.size(), len(seq_len))
                en_hi, de_out = model(input_tensor, seq_len)
                cla_pre = None
            else:
                en_hi, de_out, cla_pre = model(input_tensor, seq_len)
            label_list_train[start:start + step] = np.asarray(label)
            label_train_semi[start:start + step] = semi
            hidden_train_tmp[start:start + step, :] = en_hi[0, :, :].detach()
            start = start + step

    for isample in range(repeat):
        start = isample * eval_len
        for ith, (ith_data, seq_len, label, semi,_) in enumerate(data_eval):
            step = ith_data.shape[0]
            input_tensor = ith_data.to(device)
            if alpha == 0:
                en_hi, de_out = model(input_tensor, seq_len)
                cla_pre = None
            else:
                en_hi, de_out, cla_pre = model(input_tensor, seq_len)
            label_list_eval[start:start + step] = np.asarray(label)
            label_eval_semi[start:start + step] = semi
            hidden_eval_tmp[start:start + step, :] = en_hi[0, :, :].detach()
            start = start + step

    return hidden_train_tmp.cpu().numpy(), hidden_eval_tmp.cpu().numpy(), label_list_train.tolist(), label_list_eval.tolist(), label_train_semi, label_eval_semi
