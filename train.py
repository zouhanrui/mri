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

####################################################
####              HYPER-PARAMETERS               ###
####################################################

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind; # 0,1,2,3


# Because we have real & imaginary part of our input, data_channels is set to 2
data_channels = 1
truth_channels = 1

####################################################
####                DATA LOADING                 ###
####################################################

"""
	here loads all the data we need for training and validating.

"""

#-- Training Data --#
#data_mat = spio.loadmat('datasets/traOb.mat', squeeze_me=True)
#truths_mat = spio.loadmat('datasets/traGt.mat', squeeze_me=True)
#data = data_mat['tra']
#data = preprocess(data, data_channels)    # 4 dimension -> 3 dimension if you do data[:,:,:,1]
#truths = preprocess(truths_mat['tra'], truth_channels)

#mat v 7.3 needs HDF reader
data_mat = h5py_mat2npy('datasets/traOb.mat') # [2040, 320, 320, 1]
truths_mat = h5py_mat2npy('datasets/traGt.mat') # [2040, 320, 320, 1]
data_provider = image_util.SimpleDataProvider(data_mat, truths_mat)

#-- Validating Data --#
#vdata_mat = spio.loadmat('datasets/valOb.mat', squeeze_me=True)
#vtruths_mat = spio.loadmat('datasets/valOb.mat', squeeze_me=True)
#vdata = vdata_mat['val']
#vdata = preprocess(vdata, data_channels)
#vtruths = preprocess(vtruths_mat['val'], truth_channels)

#mat v 7.3 needs HDF reader
vdata_mat = h5py_mat2npy('datasets/valOb.mat') # [30, 320, 320, 1]
vtruths_mat = h5py_mat2npy('datasets/valGt.mat') # [30, 320, 320, 1]
valid_provider = image_util.SimpleDataProvider(vdata_mat, vtruths_mat)


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
	"summaries": True
}

net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost="mean_squared_error", **kwargs)


####################################################
####                 TRAINING                    ###
####################################################

# args for training
batch_size = 1  # batch size for training
valid_size = 1  # batch size for validating
optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'

# output paths for results
output_path = 'gpu' + gpu_ind + '/models'
prediction_path = 'gpu' + gpu_ind + '/validation'
# restore_path = 'gpu001/models/50099_cpkt'

# optional args
opt_kwargs = {
		'learning_rate': 0.001
}

# make a trainer for scadec
trainer = Trainer_bn(net, batch_size=batch_size, optimizer = "adam", opt_kwargs=opt_kwargs)
path = trainer.train(data_provider, output_path, valid_provider, valid_size, training_iters=100, epochs=1000, display_step=20, save_epoch=100, prediction_path=prediction_path)





