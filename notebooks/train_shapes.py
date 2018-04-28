
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# ### Notes from implementation
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 
# 

# In[1]:


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
import pprint
import keras.backend as KB
sys.path.append('../')

import mrcnn.model     as modellib
import mrcnn.visualize as visualize
import mrcnn.shapes    as shapes
from mrcnn.config      import Config
from mrcnn.model       import log
from mrcnn.dataset     import Dataset 

from mrcnn.utils       import stack_tensors, stack_tensors_3d
from mrcnn.datagen     import data_generator, load_image_gt
from mrcnn.callbacks   import get_layer_output_1,get_layer_output_2
from mrcnn.visualize   import plot_gaussian
# from mrcnn.pc_layer    import PCTensor
# from mrcnn.pc_layer   import PCNLayer

# Root directory of the project
ROOT_DIR = os.getcwd()
MODEL_PATH = 'E:\Models'
# Directory to save logs and trained model
MODEL_DIR = os.path.join(MODEL_PATH, "mrcnn_logs")
# Path to COCO trained weights
COCO_MODEL_PATH   = os.path.join(MODEL_PATH, "mask_rcnn_coco.h5")
RESNET_MODEL_PATH = os.path.join(MODEL_PATH, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

print("Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))
pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4)


# Build configuration object -----------------------------------------------
config = shapes.ShapesConfig()
config.BATCH_SIZE      = 5                  # Batch size is 2 (# GPUs * images/GPU).
config.IMAGES_PER_GPU  = 5                  # Must match BATCH_SIZE
config.STEPS_PER_EPOCH = 2
config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]
config.display() 

# Build shape dataset        -----------------------------------------------
# Training dataset
# generate 500 shapes 
dataset_train = shapes.ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = shapes.ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

try :
    del model, train_generator, val_generator, mm
    gc.collect()
except: 
    pass
# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 3)
# for image_id in [3]:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
print(' COCO Model Path       : ', COCO_MODEL_PATH)
print(' Checkpoint folder Path: ', MODEL_DIR)
print(MODEL_PATH)
print(COCO_MODEL_PATH)
print(RESNET_MODEL_PATH)
print(MODEL_DIR)


# # Create Model

# In[2]:


try :
    del model
    gc.collect()
except: 
    pass

model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

print('\n Outputs: ') 
pp.pprint(model.keras_model.outputs)
print('\n Losses (model.metrics_names): ') 
pp.pprint(model.keras_model.metrics_names)


# model.keras_model.summary(line_length = 150) 
# In[ ]:


#model.keras_model.summary(line_length = 120) 
# model.compile_only(learning_rate=config.LEARNING_RATE, layers='heads')
# KB.set_learning_phase(1)

# print(model.find_last())
# # Which weights to start with?
# init_with = "last"  # imagenet, coco, or last

# if init_with == "imagenet":
# #     loc=model.load_weights(model.get_imagenet_weights(), by_name=True)
    # loc=model.load_weights(RESNET_MODEL_PATH, by_name=True)
# elif init_with == "coco":
    # # Load weights trained on MS COCO, but skip layers that
    # # are different due to the different number of classes
    
    # # See README for instructions to download the COCO weights
    # loc=model.load_weights(COCO_MODEL_PATH, by_name=True,
                       # exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                # "mrcnn_bbox", "mrcnn_mask"])
# elif init_with == "last":
    # # Load the last model you trained and continue training
    # loc= model.load_weights(model.find_last()[1], by_name=True)
# print('Load weights complete')


# ###  Print some model information 

# In[ ]:


