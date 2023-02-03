#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os

from skimage.measure import block_reduce

from scipy.interpolate import interp2d

from random import randrange

class WindDataset(Dataset):
    def __init__(self, data_dir: str, mode: str = "train", lr_size: int = 32, scale_factor: int = 4, data_mean: list = [-0.77991901, -0.58770425], data_std_dev: list = [5.46725844, 5.07016567]):
        """
        data_dir: the root directory containing the "train", "val", and "test" folders.
        mode: "train", "val", or "test"
        lr_size: height/width of low res model inputs 
        scale_factor: amount to upscale by
        data_mean: mean of training data along each dimension
        data_std_dev: standard deviation of training data along each dimension
        """
        self.lr_shape = (lr_size, lr_size)
        self.hr_shape = (lr_size*scale_factor, lr_size*scale_factor)
        self.scale_factor = scale_factor
        
        self.normalize = False
        if data_mean is not None:
            self.data_mean = np.array(data_mean)
            self.data_std_dev = np.array(data_std_dev)
            self.normalize = True
            
        data_dir = os.path.join(data_dir, mode)
        self.data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        
    def normalize_arr(self, arr):
        return (arr - self.data_mean[:, None, None])/self.data_std_dev[:, None, None]
        
    def __getitem__(self, idx):
        data = np.load(self.data_files[idx]) # 2 x h x w
        region_x = randrange(0,data.shape[1]-self.hr_shape[0])
        region_y = randrange(0,data.shape[2]-self.hr_shape[1])
        region = data[:,region_x:region_x+self.hr_shape[0],region_y:region_y + self.hr_shape[1]]
        lr_region = block_reduce(region, (1, self.scale_factor, self.scale_factor), np.mean)
        
        x_vals = np.arange((self.scale_factor-1)/2, self.hr_shape[0], self.scale_factor)
        y_vals = np.arange((self.scale_factor-1)/2, self.hr_shape[1], self.scale_factor)
        
        upsampled_region = np.zeros(region.shape)
        for channel_idx in range(lr_region.shape[0]):
            interpolation_fxn = interp2d(x_vals, y_vals, lr_region[channel_idx], kind='cubic')
            upsampled_region[channel_idx] = interpolation_fxn(np.arange(self.hr_shape[0]), np.arange(self.hr_shape[1]))
        
        if self.normalize:
            # return self.normalize_arr(lr_region), self.normalize_arr(upsampled_region), self.normalize_arr(region) 
            return {'LR': self.normalize_arr(lr_region), 'HR': self.normalize_arr(upsampled_region), 'SR': self.normalize_arr(region), 'Index': idx}
        return {'LR': lr_region, 'HR': upsampled_region, 'SR': region, 'Index': idx}
        #     return {'HR': self.normalize_arr(region), 'SR': self.normalize_arr(upsampled_region), 'Index': idx}
        # return {'HR': region, 'SR': upsampled_region, 'Index': idx}
    
    def __len__(self):
        return len(self.data_files)