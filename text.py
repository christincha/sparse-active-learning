# import numpy as np
# import matplotlib.pyplot as plt
# from torch.optim import SGD
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from siamesa.siamesa_model import FCN
# a = np.loadtxt('./output/FSfewPCICA0.00_P100_en3_hid1024_orL1.txt')
# plt.plot(a[:,1], a[:,6])
# plt.show()
#
# model = FCN()
# optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
# scheduler = CosineAnnealingLR(optimizer, 10)
# for epoch in range(30):
#     for i in range(10):
#         scheduler.step()
#         print(i, optimizer.param_groups[0]['lr'])

# transform label
import json
print('a')
with open('run-.-tag-selected_sample_trainadd label.json', 'r') as f:
    sample = json.load(f)
    print(sample)