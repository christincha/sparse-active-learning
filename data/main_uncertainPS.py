from __future__ import unicode_literals, print_function, division

from Model import *
from train_uncertain import *

# load file

from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from data_loader import *
import torch
torch.cuda.set_device(0)
threshold = 0.0
drop = 0.3
# parameters
ProjectFolderName = 'NTUProject'
root_path = '/home/ws2/Documents/jingyuan/'
train_data = 'NTUtrain_cs_full.h5'
test_data = 'NTUtest_cs_full.h5'
# hyperparameter
labeled_pr = 0.05
feature_length = 75
hidden_size = 1024
batch_size = 64
en_num_layers = 3
de_num_layers = 1
middle_size = 125
cla_num_layers = 1
learning_rate = 0.0001
epoch = 300
cla_dim = [60]
k = 2
sample = 'BCE'
semi_name = './labels/base_semiLabel.npy'
uncertain_file = './labels/ps_prob_full_rp30_drop%.2f.npy'% (drop)
train_file = os.path.join(root_path, ProjectFolderName, train_data)
dataset_train =UncertaintyData(train_file, uncertain_file, semi_name, sample=sample)
dataset_test =  MyInterTrainDataset(os.path.join(root_path, ProjectFolderName, test_data), [])

shuffle_dataset = True
dataset_size_train = len(dataset_train)
dataset_size_test = len(dataset_test)

indices_train = list(range(dataset_size_train))
indices_test = list(range(dataset_size_test))
random_seed = 11111
if shuffle_dataset :
  np.random.seed(random_seed)
  np.random.shuffle(indices_train)
  np.random.shuffle(indices_test)

print("training data length: %d, validation data length: %d" % (len(indices_train), len(indices_test)))
# seperate train and validation
train_sampler = SubsetRandomSampler(indices_train)
valid_sampler = SubsetRandomSampler(indices_test)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                           sampler=train_sampler, collate_fn=pad_collate_iter)
eval_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                               sampler=valid_sampler, collate_fn=pad_collate_iter)


# 0 non labeled class
print_every = 1

model = SemiSeq(feature_length, hidden_size, cla_dim).to(device)

with torch.no_grad():
  for child in list(model.children()):
    print(child)
    for param in list(child.parameters()):
      # if param.dim() == 2:
      #   nn.init.xavier_uniform_(param)
        nn.init.uniform_(param, a=-0.05, b=0.05)

optimizer= optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

plot_pca = False
data_pca = False
predict = False
loss_type = 'L1'  # 'L1'

if loss_type == 'MSE':
    criterion_seq = nn.MSELoss(reduction='none')

if loss_type == 'L1':
    criterion_seq = nn.L1Loss(reduction='none')

criterion_cla = nn.CrossEntropyLoss(reduction='none')

past_acc = 0.4

file_output = open('./output/Pes%d_S%s_en%d_hid%d_T%.2f.txt' % (
    labeled_pr * 100, sample, en_num_layers, hidden_size, threshold), 'w')
print(' percentage of labeled %.2f' % (labeled_pr))
training(epoch, train_loader, eval_loader,
         model, optimizer,  criterion_cla, k, file_output, network='Pes', percentage=labeled_pr, en_num_l=en_num_layers, hid_s=hidden_size)

