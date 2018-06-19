# coding: utf-8
# # Mask R-CNN - Train modified model on Shapes Dataset
# ### the modified model (include model_lib) does not include any mask related heads or losses 


import os
import sys
import math
import gc
import time
import numpy as np
import argparse
import platform
import tensorflow as tf
import keras
import keras.backend as KB
sys.path.append('../')

import mrcnn.model_mod  as modellib
import mrcnn.visualize  as visualize
import mrcnn.new_shapes     as shapes
from mrcnn.config       import Config
from mrcnn.dataset      import Dataset 
from mrcnn.utils        import log, stack_tensors, stack_tensors_3d
from mrcnn.datagen      import data_generator, load_image_gt
from mrcnn.callbacks    import get_layer_output_1,get_layer_output_2
# from mrcnn.visualize    import plot_gaussian
from mrcnn.prep_notebook import prep_oldshapes_train, load_model

import pprint
pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)
print(sys.argv)
DEFAULT_LOGS_DIR = 'mrcnn_logs' 

##------------------------------------------------------------------------------------
## process input arguments
##  example:
##           train-shapes_gpu --epochs 12 --steps-in-epoch 7 --last_epoch 1234 --logs_dir mrcnn_logs
##------------------------------------------------------------------------------------
# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')
# parser.add_argument("command",
                    # metavar="<command>",
                    # help="'train' or 'evaluate' on MS COCO")
# parser.add_argument('--dataset', required=True,
                    # metavar="/path/to/coco/",
                    # help='Directory of the MS-COCO dataset')
# parser.add_argument('--limit', required=False,
                    # default=500,
                    # metavar="<image count>",
                    # help='Images to use for evaluation (defaults=500)')
                    
parser.add_argument('--model', required=False,
                    default='last',
                    metavar="/path/to/weights.h5",
                    help="'coco' , 'init' , or Path to weights .h5 file ")

parser.add_argument('--logs_dir', required=True,
                    default=DEFAULT_LOGS_DIR,
                    metavar="/path/to/logs/",
                    help='Logs and checkpoints directory (default=logs/)')
                    
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
                    
parser.add_argument('--steps_in_epoch', required=False,
                    default=1,
                    metavar="<steps in each epoch>",
                    help='Number of batches to run in each epochs (default=5)')
                    
parser.add_argument('--batch_size', required=False,
                    default=5,
                    metavar="<batch size>",
                    help='Number of data samples in each batch (default=5)')                    
                    
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
import platform
syst = platform.system()
if syst == 'Windows':
    # Root directory of the project
    print(' windows ' , syst)
    # WINDOWS MACHINE ------------------------------------------------------------------
    ROOT_DIR          = "E:\\"
    MODEL_PATH        = os.path.join(ROOT_DIR, "models")
    DATASET_PATH      = os.path.join(ROOT_DIR, 'MLDatasets')
    MODEL_DIR         = os.path.join(MODEL_PATH, args.logs_dir)
    COCO_MODEL_PATH   = os.path.join(MODEL_PATH, "mask_rcnn_coco.h5")
    DEFAULT_LOGS_DIR  = os.path.join(MODEL_PATH, "mrcnn_coco_logs")
    COCO_DATASET_PATH = os.path.join(DATASET_PATH,"coco2014")
    RESNET_MODEL_PATH = os.path.join(MODEL_PATH, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
elif syst == 'Linux':
    print(' Linx ' , syst)
    # LINUX MACHINE ------------------------------------------------------------------
    ROOT_DIR          = os.getcwd()
    MODEL_PATH        = os.path.expanduser('~/models')
    DATASET_PATH      = os.path.expanduser('~/MLDatasets')
    MODEL_DIR         = os.path.join(MODEL_PATH, args.logs_dir)
    COCO_MODEL_PATH   = os.path.join(MODEL_PATH, "mask_rcnn_coco.h5")
    COCO_DATASET_PATH = os.path.join(DATASET_PATH,"coco2014")
    DEFAULT_LOGS_DIR  = os.path.join(MODEL_PATH, "mrcnn_coco_logs")
    RESNET_MODEL_PATH = os.path.join(MODEL_PATH, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
else :
    raise Error('unreconized system  '      )


print("Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))
import pprint
    
##------------------------------------------------------------------------------------
## setup tf session and debugging 
##------------------------------------------------------------------------------------
# keras_backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))
# if 'tensorflow' == KB.backend():
#     from tensorflow.python import debug as tf_debug
#
#    config = tf.ConfigProto(device_count = {'GPU': 0} )
#    tf_sess = tf.Session(config=config)    
#    tf_sess = tf_debug.LocalCLIDebugWrapperSession(tf_sess)
#    KB.set_session(tf_sess)
#
#
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
config = shapes.NewShapesConfig()
config.BATCH_SIZE      = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
config.IMAGES_PER_GPU  = int(args.batch_size)                  # Must match BATCH_SIZE
config.STEPS_PER_EPOCH = int(args.steps_in_epoch)
config.LEARNING_RATE   = float(args.lr)

config.EPOCHS_TO_RUN   = int(args.epochs)
config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]
config.LAST_EPOCH_RAN  = int(args.last_epoch)
config.display() 

##------------------------------------------------------------------------------------
## Build shape dataset        
##------------------------------------------------------------------------------------
# Training dataset
# generate 500 shapes 
dataset_train = shapes.NewShapesDataset()
dataset_train.load_shapes(10000, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = shapes.NewShapesDataset()
dataset_val.load_shapes(2500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

##------------------------------------------------------------------------------------    
## Load and display random samples
##------------------------------------------------------------------------------------
# image_ids = np.random.choice(dataset_train.image_ids, 3)
# for image_id in [3]:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

##------------------------------------------------------------------------------------
## Build Model
##------------------------------------------------------------------------------------

try :
    del model
    gc.collect()
except: 
    pass
KB.clear_session()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR, FCN_layers = True)

print(' COCO Model Path       : ', COCO_MODEL_PATH)
print(' Checkpoint folder Path: ', MODEL_DIR)
print(' Model Parent Path     : ', MODEL_PATH)
# print(model.find_last())

##----------------------------------------------------------------------------------------------
## Load Model Weight file
##----------------------------------------------------------------------------------------------
load_model(model, init_with = args.model)   

config.display()  
model.layer_info()

##----------------------------------------------------------------------------------------------
##  Training
## 
## Train in two stages:
## 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly 
##    initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). 
##    To train only the head layers, pass `layers='heads'` to the `train()` function.
## 
## 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to 
##    show the process. Simply pass `layers="all` to train all layers.
## ## Training head using  Keras.model.fit_generator()
##----------------------------------------------------------------------------------------------
print(config.BATCH_SIZE)
print(model.config.BATCH_SIZE)



##----------------------------------------------------------------------------------------------
## Train the head branches
## Passing layers="heads" freezes all layers except the head
## layers. You can also pass a regular expression to select
## which layers to train by name pattern.
##----------------------------------------------------------------------------------------------
## Last run prior to FCN training was 3699, last checkpoint was 3892  ...start at 3899

train_layers = [ 'mrcnn', 'fpn','rpn']
loss_names   = [ "rpn_class_loss", "rpn_bbox_loss" , "mrcnn_class_loss", "mrcnn_bbox_loss"]
model.epoch                  = config.LAST_EPOCH_RAN
model.config.LEARNING_RATE   = config.LEARNING_RATE
model.config.STEPS_PER_EPOCH = config.STEPS_PER_EPOCH

model.train(dataset_train, dataset_val, 
            learning_rate = model.config.LEARNING_RATE, 
            epochs_to_run = config.EPOCHS_TO_RUN ,
#             epochs = 25,            
#             batch_size = 0
#             steps_per_epoch = 0 
            layers = train_layers,
            losses = loss_names,
            min_LR = 1.0e-6,
            )
            

##----------------------------------------------------------------------------------------------
## Train the FCN only 
## Passing layers="heads" freezes all layers except the head
## layers. You can also pass a regular expression to select
## which layers to train by name pattern.
##----------------------------------------------------------------------------------------------            
"""
train_layers = ['fcn']
loss_names   = ["fcn_norm_loss"]

model.epoch                  = config.LAST_EPOCH_RAN
model.config.LEARNING_RATE   = config.LEARNING_RATE
model.config.STEPS_PER_EPOCH = config.STEPS_PER_EPOCH

model.train(dataset_train, dataset_val, 
            learning_rate = model.config.LEARNING_RATE, 
            epochs_to_run = config.EPOCHS_TO_RUN,
#             epochs = 25,            # total number of epochs to run (accross multiple trainings)
            layers = train_layers,
            losses = loss_names,
            min_LR = 1.0e-7,
            )
"""