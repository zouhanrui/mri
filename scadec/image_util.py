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
    
    def __call__(self, fix, batch_size, itr, num):
        if type(batch_size) == int and not fix:
            # X and Y are the images and truths
            train_data, truths = self._next_batch(batch_size, itr, num)
        elif type(batch_size) == int and fix:
            train_data, truths = self._fix_batch(batch_size)
        elif type(batch_size) == str and batch_size == 'full':
            train_data, truths = self._full_batch() 
        else:
            raise ValueError("Invalid batch_size: "%batch_size)
        
        return train_data, truths

    def _next_batch(self, batch_size, itr, num):
        pass

    def _full_batch(self):
        pass


class SimpleDataProvider(BaseDataProvider):
    
    def __init__(self, data, truths, patch_shape, patch_overlap_rate, num_patches):
        super(SimpleDataProvider, self).__init__()
        self.data = np.float64(data)
        self.truths = np.float64(truths)
        self.img_channels = self.data.shape[4]
        self.truth_channels = self.truths.shape[4]
        self.file_count = data.shape[0]     # 10 sample cubes(96*320*320)
        self.patch_shape = patch_shape      #(32,32,32)
        self.patch_overlap_rate = patch_overlap_rate # 0.5
        self.num_patches = num_patches # 3087
        
        
    def _get_patch_cube(self, num, data_or_truths):
        all_patches_per_cube = self._partition2patches(num, data_or_truths) # (num_patches, nd, nx, ny, channel)
        return self._process_data(all_patches_per_cube)

    def _next_batch(self, batch_size, itr, num):   
        # (1,96,320,320,1) run out of memory
        # return patch (1,32,32,32,1)
        # self.data = (10,96,320,320,1)
        #idx = np.random.choice(self.file_count, n, replace=False)
        #idx = np.random.choice(self.file_count, 1, replace=False) # Get one of 10 whole cubes
        #img = self.data[idx[0]] # (96*320*320*1)
        
        #all_patches_data = self._partition2patches(self.data) # (n,32,32,32,1)
        #all_patches_truths = self._partition2patches(self.truths)
        all_patches_data = self._get_patch_cube(num, "data")
        all_patches_truths = self._get_patch_cube(num, "truths")
        
        nd = self.patch_shape[0] #32
        nx = self.patch_shape[1] #32
        ny = self.patch_shape[2] #32
        
        X = np.zeros((batch_size, nd, nx, ny, self.img_channels))
        Y = np.zeros((batch_size, nd, nx, ny, self.truth_channels))
        X = all_patches_data[itr * batch_size : (itr * batch_size + batch_size)]
        Y = all_patches_truths[itr * batch_size : (itr * batch_size + batch_size)]
        #for i in range(batch_size):
         #   X[i] = _get_patch_data(itr * batch_size + i, num) # return shape (32, 32, 32, 1)
          #  Y[i] = _get_patch_truths(itr * batch_size + i, num)
            
        return X, Y

    def _fix_batch(self, n):
        # first n data   
        img = self.data[0] #(30*320*320*1)
        
        nd = img.shape[0]  #30
        nx = img.shape[1]  #320
        ny = img.shape[2]  #320
        X = np.zeros((n, nd, nx, ny, self.img_channels))   #(N, D, H, W, C)
        Y = np.zeros((n, nd, nx, ny, self.truth_channels)) #(N, D, H, W, C)
        
        X = self._process_data(self.data) # (1,30,320,320,1)
        Y = self._process_data(self.truths) # (1,30,320,320,1)
        
        
        #for i in range(n):
            #print(i)
            #X[i] = self._process_data(self.data[i]) # self.data:(30*320*320*1)
            #Y[i] = self._process_truths(self.truths[i])
        return X, Y

    def _full_batch(self):
        return self.data, self.truths

    def _process_truths(self, truth):
        # normalization by channels
        truth = np.clip(np.fabs(truth), self.a_min, self.a_max)
        for channel in range(self.truth_channels):
            truth[:,:,:,channel] -= np.amin(truth[:,:,:,channel])
            truth[:,:,:,channel] /= np.amax(truth[:,:,:,channel])
        return truth

    def _process_data(self, data): # input (num_patches, nd, nx, ny, channel)
        # normalization by channels
        #data = np.clip(np.fabs(data), self.a_min, self.a_max)
        for num in range(data.shape[0]):
            for channel in range(self.img_channels):
                data[num,:,:,:,channel] -= np.amin(data[num,:,:,:,channel])
                data[num,:,:,:,channel] /= np.amax(data[num,:,:,:,channel])
        return data
    
    
    def _get_patch_data(self, itr, num): 
        all_patches_data = self._partition2patches(num) # (num_patches, nd, nx, ny, channel)
        patch = all_patches_data[itr] #(32,32,32,1)
        return self._process_data(patch)

    def _get_patch_truths(self, itr, num): 
        all_patches_truths = self._partition2patches(num)
        patch = all_patches_truths[itr] #(32,32,32,1)
        return self._process_truths(patch)
    
    
    def _partition2patches(self, num, data_or_truths): # data is one single cube (96,320,320,1)
        if data_or_truths == "data":
            data = self.data[num] # (96,320,320,1)
        else:
            data = self.truths[num]
        overlap_rate = self.patch_overlap_rate   
        num_patches = self.num_patches
        patch_shape = self.patch_shape
        img_channels = self.img_channels
        
        nd_patch = patch_shape[0] #32
        nx_patch = patch_shape[1] #32
        ny_patch = patch_shape[2] #32
        nd_patch_stride = int(nd_patch * (1 - overlap_rate)) #16
        nx_patch_stride = int(nx_patch * (1 - overlap_rate)) #16
        ny_patch_stride = int(ny_patch * (1 - overlap_rate)) #16
        nd_data = data.shape[0] #96
        nx_data = data.shape[1] #320
        ny_data = data.shape[2] #320
        
        
        num_nd = int((nd_data+2*nd_patch_stride-nd_patch)/nd_patch_stride + 1) # 7
        num_nx = int((nx_data+2*nx_patch_stride-nx_patch)/nx_patch_stride + 1) # 21
        num_ny = int((ny_data+2*ny_patch_stride-ny_patch)/ny_patch_stride + 1) # 21

        
        
        all_patches_shape = (num_patches, nd_patch, nx_patch, ny_patch, img_channels)
        all_patches = np.zeros((all_patches_shape)) #(3087,32,32,32,1)
        
        pad_nd = int(data.shape[0]+2*nd_patch_stride)
        pad_nx = int(data.shape[1]+2*nx_patch_stride)
        pad_ny = int(data.shape[2]+2*ny_patch_stride)
        channel = data.shape[3]
        
        zero_padding_data = np.zeros((pad_nd, pad_nx, pad_ny, channel)) # (128,352,352,1)
        zero_padding_data[nd_patch_stride:(pad_nd-nd_patch_stride), nx_patch_stride:(pad_nx-nx_patch_stride), ny_patch_stride:(pad_ny-ny_patch_stride), :] = data
        
        
        for nd in range(num_nd):
            #print(nd)   
            for nx in range(num_nx):
                for ny in range(num_ny):
                     #print(zero_padding_data[nd*nd_patch_stride:(nd*nd_patch_stride+nd_patch), nx*nx_patch_stride:(nx*nx_patch_stride+nx_patch), ny*ny_patch_stride:(ny*ny_patch_stride+ny_patch), :].shape)
                    all_patches[nd*num_nx*num_ny+nx*num_ny+ny] = zero_padding_data[nd*nd_patch_stride:(nd*nd_patch_stride+nd_patch), nx*nx_patch_stride:(nx*nx_patch_stride+nx_patch), ny*ny_patch_stride:(ny*ny_patch_stride+ny_patch), :] # (32,32,32,1)
                    
        return all_patches # output the patches (3087,32,32,32,1) num_patches = 3087/cube