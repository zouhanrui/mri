from scadec.unet_bn import Unet_bn
from scadec.train import Trainer_bn
from scadec import util
from scadec import image_util

import scipy.io as spio
import numpy as np
import os
import sys

####################################################
####                DATA LOADING                 ###
####################################################

"""
	here loads all the data we need for training and validating.

"""

#-- Training Data --#
data_mat = spio.loadmat('data/mc_Subj01_T=400.mat', squeeze_me=True) # [320, 320, 960]
data_mat = data_mat['mcSubj01'].T   
data_mat = data_mat.swapaxes(1,2)   # (960,320,320)
data_mat = data_mat.reshape(960,320,320,1)
val_data = data_mat[0:96]
train_data = data_mat[96:960]


truths_mat = spio.loadmat('data/cs_Subj01_T=2000.mat', squeeze_me=True) # [10, 96, 320, 320, 1]
truths_mat = truths_mat['csSubj01'].T
truths_mat = truths_mat.swapaxes(1,2)
truths_mat = truths_mat.reshape(960,320,320,1)
val_truths = truths_mat[0:96]
train_truths = truths_mat[96:960]

patch_shape = (32,32,32) # tra:(864,320,320)->(27,10,10)->2700 patches. val:(96,320,320)->(3,10,10)->300 patches

data_provider = image_util.SimpleDataProvider(train_data, train_truths, patch_shape)
valid_provider = image_util.SimpleDataProvider(val_data, val_truths, patch_shape)

####################################################
####              HYPER-PARAMETERS               ###
####################################################

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind; # 0,1,2,3

data_channels = 1
truth_channels = 1

batch_size = 18
valid_size = 18

####################################################
####                 TRAINING                    ###
####################################################

training_iters_per_epoch = int(2700 / batch_size)

optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'

# output paths for results
output_path = 'gpu' + gpu_ind + '/models'
prediction_path = 'gpu' + gpu_ind + '/validation'
# restore_path = 'gpu001/models/50099_cpkt'

# optional args
opt_kwargs = {
		'learning_rate': 0.0001
}

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
path = trainer.train(data_provider, output_path, valid_provider, valid_size, training_iters_per_epoch, epochs=50, display_step=10, save_epoch=100, prediction_path=prediction_path)





