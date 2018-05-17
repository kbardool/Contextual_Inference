
# coding: utf-8

# # Mask R-CNN - Train modified model on Shapes Dataset
# 
# ### the modified model (include model_lib) does not include any mask related heads or losses 


import os
import sys
import random
import math
import re
import  gc
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.backend as KB
sys.path.append('../')

import mrcnn.model_mod as modellib
import mrcnn.visualize as visualize
import mrcnn.shapes    as shapes
from mrcnn.config      import Config
from mrcnn.dataset     import Dataset 
from mrcnn.utils       import log, stack_tensors, stack_tensors_3d
from mrcnn.datagen     import data_generator, load_image_gt
from mrcnn.callbacks   import get_layer_output_1,get_layer_output_2
from mrcnn.visualize   import plot_gaussian

##------------------------------------------------------------------------------------
## process input arguments
#  call example train-shapes_gpu --epochs 12 --steps-in-epoch 5
##------------------------------------------------------------------------------------
# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')
# parser.add_argument("command",
                    # metavar="<command>",
                    # help="'train' or 'evaluate' on MS COCO")
# parser.add_argument('--dataset', required=True,
                    # metavar="/path/to/coco/",
                    # help='Directory of the MS-COCO dataset')
parser.add_argument('--model', required=False,
                    default='last',
                    metavar="/path/to/weights.h5",
                    help="'coco' , 'init' , or Path to weights .h5 file ")
# parser.add_argument('--logs', required=False,
                    # default=DEFAULT_LOGS_DIR,
                    # metavar="/path/to/logs/",
                    # help='Logs and checkpoints directory (default=logs/)')
# parser.add_argument('--limit', required=False,
                    # default=500,
                    # metavar="<image count>",
                    # help='Images to use for evaluation (defaults=500)')
parser.add_argument('--last_epoch', required=False,
                    default=0,
                    metavar="<last epoch ran>",
                    help='Identify last completed epcoh for tensorboard continuation')

parser.add_argument('--lr', required=False,
                    default=0.001,
                    metavar="<learning rate>",
                    help='Learning Rate (default=0.001)')

parser.add_argument('--epochs', required=False,
                    default=3,
                    metavar="<epochs to run>",
                    help='Number of epochs to run (default=3)')
                    
parser.add_argument('--steps_per_epoch', required=False,
                    default=1,
                    metavar="<steps in each epoch>",
                    help='Number of batches to run in each epochs (default=5)')
                    
args = parser.parse_args()
# args = parser.parse_args("train --dataset E:\MLDatasets\coco2014 --model mask_rcnn_coco.h5 --limit 10".split())
pp.pprint(args)
print("Model              :   ", args.model)
# print("Dataset: ", args.dataset)
# print("Logs:    ", args.logs)
# print("Limit:   ", args.limit)
print("Epochs to run      :   ", args.epochs)
print("Steps in each epoch:   ", args.steps_in_epoch)


##------------------------------------------------------------------------------------
## setup project directories
#---------------------------------------------------------------------------------
# # Root directory of the project 
# MODEL_DIR    :    Directory to save logs and trained model
# COCO_MODEL_PATH  : Path to COCO trained weights
#---------------------------------------------------------------------------------

# WINDOWS MACHINE ------------------------------------------------------------------
# WINDOWS MACHINE ------------------------------------------------------------------
ROOT_DIR          = 'E:\'
MODEL_PATH        = os.path.join(ROOT_DIR, "models")
DATASET_PATH      = os.path.join(ROOT_DIR, 'MLDatasets')
#### MODEL_DIR         = os.path.join(MODEL_PATH, "mrcnn_logs")
COCO_MODEL_PATH   = os.path.join(MODEL_PATH, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR  = os.path.join(MODEL_PATH, "mrcnn_coco_logs")
COCO_DATASET_PATH = os.path.join(DATASET_PATH,"coco2014")

# RESNET_MODEL_PATH = os.path.join(MODEL_PATH, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

# LINUX MACHINE ------------------------------------------------------------------
# ROOT_DIR          = os.getcwd()
# MODEL_PATH        = os.path.expanduser('~/models')
# DATASET_PATH      = os.path.expanduser('~/MLDatasets')
#### MODEL_DIR         = os.path.join(MODEL_PATH, "mrcnn_logs")
# COCO_MODEL_PATH   = os.path.join(MODEL_PATH, "mask_rcnn_coco.h5")
# COCO_DATASET_PATH = os.path.join(DATASET_PATH,"coco2014")
# DEFAULT_LOGS_DIR = os.path.join(MODEL_PATH, "mrcnn_coco_logs")
# RESNET_MODEL_PATH = os.path.join(MODEL_PATH, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

##------------------------------------------------------------------------------------
## setup tf session and debugging 
##------------------------------------------------------------------------------------
   # keras_backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))

# if 'tensorflow' == KB.backend():
    # from tensorflow.python import debug as tf_debug

    # config = tf.ConfigProto(
            # device_count = {'GPU': 0}
        # )
    # tf_sess = tf.Session(config=config)    
    # tf_sess = tf_debug.LocalCLIDebugWrapperSession(tf_sess)
    # KB.set_session(tf_sess)


#   tfconfig = tf.ConfigProto(
#               gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
#               device_count = {'GPU': 1}
#              )    
#     tfconfig = tf.ConfigProto()
#     tfconfig.gpu_options.allow_growth=True
#     tfconfig.gpu_options.visible_device_list = "0"
#     tfconfig.gpu_options.per_process_gpu_memory_fraction=0.5
#     tf_sess = tf.Session(config=tfconfig)
#     set_session(tf_sess)
##------------------------------------------------------------------------------------


##------------------------------------------------------------------------------------
## Build configuration object 
##------------------------------------------------------------------------------------
config = shapes.ShapesConfig()
config.BATCH_SIZE      = 5                  # Batch size is 2 (# GPUs * images/GPU).
config.IMAGES_PER_GPU  = 5                  # Must match BATCH_SIZE
config.STEPS_PER_EPOCH = int(args.steps_in_epoch)
config.LEARNING_RATE   = float(args.lr)

config.EPOCHS_TO_RUN   = int(args.epochs)
config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]
config.LAST_EPOCH_RAN  = int(args.last_epoch)
# config.LAST_EPOCH_RAN  = 5784
config.display() 

##------------------------------------------------------------------------------------
## Build shape dataset        
##------------------------------------------------------------------------------------
# Training dataset
# generate 500 shapes 
dataset_train = shapes.ShapesDataset()
dataset_train.load_shapes(2000, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = shapes.ShapesDataset()
dataset_val.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

##------------------------------------------------------------------------------------    
## Load and display random samples
##------------------------------------------------------------------------------------
# image_ids = np.random.choice(dataset_train.image_ids, 3)
# for image_id in [3]:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

import pprint
pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4)
##------------------------------------------------------------------------------------
## Build Model
##------------------------------------------------------------------------------------

try :
    del model
    gc.collect()
except: 
    pass
KB.clear_session()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)



print(' COCO Model Path       : ', COCO_MODEL_PATH)
print(' Checkpoint folder Path: ', MODEL_DIR)
print(' Model Parent Path     : ', MODEL_PATH)
# print(' Resent Model Path     : ', RESNET_MODEL_PATH)
print(model.find_last())

# model.compile_only(learning_rate=config.LEARNING_RATE, layers='all')
# tst = model.keras_model.to_json()
# save_model(MODEL_DIR, 'my_saved_model')
# print(model.find_last())
#model.keras_model.summary(line_length = 120) 

##------------------------------------------------------------------------------------
## Load Model hf5 file 
##------------------------------------------------------------------------------------
# KB.set_learning_phase(1)
'''
methods to load weights
1 - load a specific file
2 - find a last checkpoint in a specific folder 
3 - use init_with keyword 
'''
## 1- look for a specific weights file 
## Load trained weights (fill in path to trained weights here)
# model_path  = 'E:\\Models\\mrcnn_logs\\shapes20180428T1819\\mask_rcnn_shapes_5784.h5'
# print(' model_path : ', model_path )
# assert model_path != "", "Provide path to trained weights"
# print("Loading weights from ", model_path)
# model.load_weights(model_path, by_name=True)    
# print('Load weights complete')

# ## 2- look for last checkpoint file in a specific folder (not working correctly)
# model.config.LAST_EPOCH_RAN = 5784
# model.model_dir = 'E:\\Models\\mrcnn_logs\\shapes20180428T1819'
# last_model_found = model.find_last()
# print(' last model in MODEL_DIR: ', last_model_found)
# # loc= model.load_weights(model.find_last()[1], by_name=True)
# # print('Load weights complete :', loc)


## 3- Use init_with keyword
## Which weights to start with?
init_with = args.model  # imagenet, coco, or last

if init_with == "imagenet":
#     loc=model.load_weights(model.get_imagenet_weights(), by_name=True)
    loc=model.load_weights(RESNET_MODEL_PATH, by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    loc=model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
# Load the last model you trained and continue training                               
elif init_with == "last":
    lastChkPointFile = model.find_last()[1]
    print('    Last Checkpoint file output: ', lastChkPointFile)
    loc= model.load_weights(lastChkPointFile, by_name=True)

print()
# print("Dataset: ", args.dataset)
# print("Logs:    ", args.logs)
# print("Limit:   ", args.limit)
print("    Model                 : ", args.model)
print("    Last Epcoh Ran        : ", config.LAST_EPOCH_RAN)
print("    Epochs to run         : ", config.EPOCHS_TO_RUN)
print("    Steps in each epoch   : ", config.STEPS_PER_EPOCH)
print("    Execution resumes from epoch: ", model.epoch)
print()
print('    Root dir              : ', ROOT_DIR)
print('    Model path            : ', MODEL_PATH)
print('    Model dir             : ', MODEL_DIR)
print('    COCO Model Path       : ', COCO_MODEL_PATH)
print('    Resnet Model Path     : ', RESNET_MODEL_PATH)
print('    Checkpoint folder Path: ', MODEL_DIR)

config.display() 

##------------------------------------------------------------------------------------
## Training heads using fit_generator()
##------------------------------------------------------------------------------------

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

# model.train(dataset_train, dataset_val, 
#             learning_rate=config.LEARNING_RATE, 
# #             epochs = 69,
#             epochs_to_run =2, 
#             layers='heads')

train_layers = ['mrcnn', 'fpn','rpn']
loss_names   = [  "rpn_class_loss", "rpn_bbox_loss" , "mrcnn_class_loss", "mrcnn_bbox_loss"]
 
config.LEARNING_RATE = 1.0e-4

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
#             epochs = 25,
            epochs_to_run =2000, 
#             batch_size = 0
#             steps_per_epoch = 0 
            layers = train_layers,
            losses = loss_names,
            min_LR = 1.0e-6
            )
