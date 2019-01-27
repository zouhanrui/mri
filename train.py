from scadec.unet_bn import Unet_bn
from scadec.train import Trainer_bn

from scadec import image_util

import scipy.io as spio
import numpy as np
import os

import sys
import h5py
####################################################
####                 FUNCTIONS                   ###
####################################################

# make the data a 4D vector
def preprocess(data, channels):
	nx = data.shape[1]
	ny = data.shape[2]
	return data.reshape((-1, nx, ny, channels))


def h5py_mat2npy(datemat):
    print('Loading '+datemat)
    f = h5py.File(datemat)
    #test=a[a.keys()[i]]
    #test=a['train_nbm_5']
    
    for data in f:
        test=np.array(f[data])
        test=test.T 
        
        if len(np.shape(test)) == 3:
            nx,ny = np.shape(test)[1:]
            chs = 1
        elif len(np.shape(test)) == 4:
            nx,ny, chs = np.shape(test)[1:]
            
    test_x = np.reshape(test,[-1,nx,ny,chs])
    return test_x


def my_h5py_mat2npy(datamat):
    print('Loading '+datamat)
    f = h5py.File(datamat)
    for data in f:
        npy = np.array(f[data])

    npy = npy.T #(N, H, W)
    if npy.shape[0] > 96:
        npy = npy.reshape(-1,204,320,320,1)
    else:
        npy = npy.reshape(-1,npy.shape[0], 320, 320, 1)
    print(npy.shape) # (N, D, H, W, C)
    return npy    


####################################################
####                DATA LOADING                 ###
####################################################

"""
	here loads all the data we need for training and validating.

"""

#-- Training Data --#
data_mat = spio.loadmat('data/mc_Subj01_T=400.mat', squeeze_me=True) # [10, 96, 320, 320, 1]
data_mat =data_mat['mcSubj01'].reshape(10, 96, 320, 320, 1)
truths_mat = spio.loadmat('data/cs_Subj01_T=2000.mat', squeeze_me=True) # [10, 96, 320, 320, 1]
truths_mat = truths_mat['csSubj01'].reshape(10,96,320,320,1)

#-- Validating Data --#
vdata_mat = my_h5py_mat2npy('datasets/valOb.mat') # [1, 30, 320, 320, 1]
vtruths_mat = my_h5py_mat2npy('datasets/valGt.mat') # [1, 30, 320, 320, 1]


####################################################
####              HYPER-PARAMETERS               ###
####################################################

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind; # 0,1,2,3


# Because we have real & imaginary part of our input, data_channels is set to 2
data_channels = 1
truth_channels = 1

# patch and overlapping
patch_overlap_rate = 0.5   # 0.75 bs=11; 0.5 bs=7
patch_shape = (32,32,32)

nd_patch = patch_shape[0] #32
nx_patch = patch_shape[1] #32
ny_patch = patch_shape[2] #32
nd_patch_stride = nd_patch * (1 - patch_overlap_rate) #16
nx_patch_stride = nx_patch * (1 - patch_overlap_rate) #16
ny_patch_stride = ny_patch * (1 - patch_overlap_rate) #16
nd_data = data_mat.shape[1] #96
nx_data = data_mat.shape[2] #320
ny_data = data_mat.shape[3] #320
num_nd = int((nd_data+2*nd_patch_stride-nd_patch)/nd_patch_stride + 1) # 7
num_nx = int((nx_data+2*nx_patch_stride-nx_patch)/nx_patch_stride + 1) # 21
num_ny = int((ny_data+2*ny_patch_stride-ny_patch)/ny_patch_stride + 1) # 21

num_patches = int(num_nd * num_nx * num_ny) # 3087 Numbers of patches per big chunk of cube




# Always use the smallest dimension among nd,nx,ny to get the smallest possible batch_size for memory
# Batch size of training and validating
batch_size = int(1 + (nd_data + 2*nd_patch_stride - nd_patch)/nd_patch_stride) # tra 1+(96+2*16-32)/16 = 7
valid_size = 1  # val

####################################################
####                 TRAINING                    ###
####################################################

# args for training
training_iters_per_cube = int(num_patches / batch_size) # Numbers of training per cube
print("Numbers of training per cube: {}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(training_iters_per_cube))
num_cubes = data_mat.shape[0] # 10 num_cubes * training_iters_per_cube = num_iters_per_epoch
optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'

# output paths for results
output_path = 'gpu' + gpu_ind + '/models'
prediction_path = 'gpu' + gpu_ind + '/validation'
# restore_path = 'gpu001/models/50099_cpkt'

# optional args
opt_kwargs = {
		'learning_rate': 0.001
}

####################################################
####                DATA_PROVIDER                ###
####################################################

data_provider = image_util.SimpleDataProvider(data_mat, truths_mat, patch_shape, patch_overlap_rate, num_patches)
valid_provider = image_util.SimpleDataProvider(vdata_mat, vtruths_mat, patch_shape, patch_overlap_rate, num_patches)

####################################################
####                  NETWORK                    ###
####################################################

"""
	here we specify the neural network.

"""

#-- Network Setup --#
# set up args for the unet
kwargs = {
	"layers": 5,           # how many resolution levels we want to have
	"conv_times": 2,       # how many times we want to convolve in each level
	"features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
	"filter_size": 3,      # filter size used in convolution
	"pool_size": 2,        # pooling size used in max-pooling
	"summaries": False
}

net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost="mean_squared_error", **kwargs)


####################################################
####             Let's have fun                  ###
####################################################

# make a trainer for scadec
trainer = Trainer_bn(net, batch_size=batch_size, optimizer = "adam", opt_kwargs=opt_kwargs)
path = trainer.train(data_provider, output_path, valid_provider, valid_size, training_iters_per_cube, num_cubes, epochs=1000, display_step=20, save_epoch=100, prediction_path=prediction_path)





