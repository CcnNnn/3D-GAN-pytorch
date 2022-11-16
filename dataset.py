import os
import scipy.ndimage as nd
import scipy.io as io
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from config import Config


opt = Config()

def getVoxelFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0,0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels


def getImagesForACategory(dir_path, category, is_train=True):
    path = os.path.join(dir_path, category, '30')
    data_dir = path + '/train__' if is_train else path + '/test'
    filenames = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith('.mat')]
    return filenames


class Getloader(Dataset):
    def __init__(self, filename):
        self.filename = filename

    def __getitem__(self, index):
        voxels = io.loadmat(self.filename[index])['instance']
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
        if cube_len != 32 and cube_len == 64:
            voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
        return np.expand_dims(voxels.astype(np.float32), 0)

    def __len__(self):
        return len(self.filename)


cube_len = 64
dir_path = r'D:\3dgan\data'
file_path = getImagesForACategory(dir_path=dir_path, category='car', is_train=True)
dataset = Getloader(file_path)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=0)

