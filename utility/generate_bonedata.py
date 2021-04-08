
from data.relic_Dataset import ThreestreamSemiDataset
import os
if __name__ == '__main__':
    root_path = '/home/ws2/Documents/jingyuan/NTUProject'
    train_path = os.path.join(root_path, 'NTUtrain_cs_full.h5')

    train_bone_path = os.path.join(root_path, 'NTUtrain_cs_bone.json')
    test_bone_path = os.path.join(root_path, 'NTUtest_cs_bone.json')
    # train bone
    train_data = ThreestreamSemiDataset(train_path,1)
    train_data.bone_transform(train_bone_path)

    # test
    # test_path = os.path.join(root_path, 'NTUtest_cs_full.h5')
    # test_data = ThreestreamSemiDataset(test_path, 1)
    # test_data.bone_transform(test_bone_path)