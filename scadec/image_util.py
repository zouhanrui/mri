# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
Modified on Feb, 2018 based on the work of jakeret

author: yusun
'''
from __future__ import print_function, division, absolute_import, unicode_literals

#import cv2
import glob
import numpy as np
import sys
from PIL import Image


class BaseDataProvider(object):

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
    
    def __call__(self, n, fix=False):
        if type(n) == int and not fix:
            # X and Y are the images and truths
            train_data, truths = self._next_batch(n)
        elif type(n) == int and fix:
            train_data, truths = self._fix_batch(n)
        elif type(n) == str and n == 'full':
            train_data, truths = self._full_batch() 
        else:
            raise ValueError("Invalid batch_size: "%n)
        
        return train_data, truths

    def _next_batch(self, n):
        pass

    def _full_batch(self):
        pass


class SimpleDataProvider(BaseDataProvider):
    
    def __init__(self, data, truths, patch_shape):
        super(SimpleDataProvider, self).__init__()
        self.data = np.float64(data)        # (864,320,320,1)
        self.truths = np.float64(truths)
        self.img_channels = self.data.shape[3]
        self.truth_channels = self.truths.shape[3]
        self.patch_shape = patch_shape      # (32,32,32)
        self.file_count = data.shape[0]
        self.all_patches_data = self._partition2patches("data")      # (2700,32,32,32,1) / (300,32,32,32,1)
        self.all_patches_truths = self._partition2patches("truths")  # (2700,32,32,32,1) / (300,32,32,32,1) 
    
    def _get_patch_cube(self, num, data_or_truths):
        all_patches_per_cube = self._partition2patches(num, data_or_truths) # (num_patches, nd, nx, ny, channel)
        #return all_patches_per_cube
        return self._process_data(all_patches_per_cube)

    def _next_batch(self, n):   
        startidx = np.random.choice(self.file_count-n, 1, replace=False)
        
        X = np.zeros((n, 32,32,32,1))
        Y = np.zeros((n, 32,32,32,1))
        for i in range(n):
            X[i] = self._process_data(self.all_patches_data[startidx+i])
            Y[i] = self._process_truths(self.all_patches_truths[startidx+i])
        return X, Y

    def _fix_batch(self, n):
        startidx = np.random.choice(self.file_count-n, 1, replace=False)

        X = np.zeros((n, 32,32,32,1))
        Y = np.zeros((n, 32,32,32,1))
        for i in range(n):
            X[i] = self._process_data(self.all_patches_data[startidx+i])
            Y[i] = self._process_truths(self.all_patches_truths[startidx+i])
        return X, Y

    def _full_batch(self):
        return self.data, self.truths

    def _process_truths(self, truth):
        # input (32,32,32,1) normalize every patch by channel
        truth = np.clip(np.fabs(truth), self.a_min, self.a_max)
        for channel in range(self.truth_channels):
            truth[:,:,:,channel] -= np.amin(truth[:,:,:,channel])
            truth[:,:,:,channel] /= np.amax(truth[:,:,:,channel])
        return truth

    def _process_data(self, data):
         # input (32,32,32,1) normalize every patch by channel
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        for channel in range(self.img_channels):
            data[:,:,:,channel] -= np.amin(data[:,:,:,channel])
            data[:,:,:,channel] /= np.amax(data[:,:,:,channel])
        return data
    
    
    def _get_patch_data(self, itr, num): 
        all_patches_data = self._partition2patches(num) # (num_patches, nd, nx, ny, channel)
        patch = all_patches_data[itr] #(32,32,32,1)
        return self._process_data(patch)

    def _get_patch_truths(self, itr, num): 
        all_patches_truths = self._partition2patches(num)
        patch = all_patches_truths[itr] #(32,32,32,1)
        return self._process_truths(patch)
    
    
    def _partition2patches(self, data_or_truths): # data is one single cube (96,320,320,1)
        if data_or_truths == "data":
            data = self.data # tra:(864,320,320,1) / val:(96,320,320,1)
        else:
            data = self.truths
         
        nd_num = int(data.shape[0] / 32)
        num_patches = int(10*10*nd_num)
        all_patches_shape = (num_patches, 32, 32, 32, 1)
        all_patches = np.zeros((all_patches_shape)) 

        for nd in range(nd_num):
            for nx in range(10):
                for ny in range(10):
                    all_patches[nd*100+nx*10+ny] = data[nd*32:(nd*32+32), nx*32:(nx*32+32), ny*32:(ny*32+32)] # (32,32,32,1)

        print(all_patches.shape)
        return all_patches # output the patches (3087,32,32,32,1) num_patches = 3087/cube
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    