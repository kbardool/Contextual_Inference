'''
prep_dev_notebook:
pred_newshapes_dev: Runs against new_shapes
'''
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
    

import mrcnn.model_mod     as modellib
 
import mrcnn.visualize as visualize
import mrcnn.shapes    as shapes
import mrcnn.new_shapes as new_shapes

from mrcnn.config      import Config
from mrcnn.model       import log
from mrcnn.dataset     import Dataset 

from mrcnn.utils       import stack_tensors, stack_tensors_3d
from mrcnn.datagen     import data_generator, load_image_gt
# from mrcnn.callbacks   import get_layer_output_1,get_layer_output_2
# from mrcnn.visualize   import plot_gaussian
# from mrcnn.pc_layer    import PCTensor
# from mrcnn.pc_layer   import PCNLayer

# Root directory of the project
ROOT_DIR = os.getcwd()
MODEL_PATH = 'E:\Models'
# Directory to save logs and trained model
# Path to COCO trained weights
COCO_MODEL_PATH   = os.path.join(MODEL_PATH, "mask_rcnn_coco.h5")
RESNET_MODEL_PATH = os.path.join(MODEL_PATH, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

print("Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))
import pprint
pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)





def prep_dev_notebook(init_with = None, FCN_layers = False):

    MODEL_DIR = os.path.join(MODEL_PATH, "mrcnn_development_logs")

    # Build configuration object -----------------------------------------------
    config = shapes.ShapesConfig()
    config.BATCH_SIZE      = 5                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU  = 5                  # Must match BATCH_SIZE
    config.STEPS_PER_EPOCH = 4
    config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]
    # config.display() 

    # Build shape dataset        -----------------------------------------------
    # Training dataset
    # generate 500 shapes 
    dataset_train = shapes.ShapesDataset()
    dataset_train.load_shapes(150, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()
     
    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR, FCN_layers = FCN_layers)

    print(' COCO Model Path       : ', COCO_MODEL_PATH)
    print(' Checkpoint folder Path: ', MODEL_DIR)
    print(' Model Parent Path     : ', MODEL_PATH)
    print(' Resent Model Path     : ', RESNET_MODEL_PATH)

    load_model(model, init_with = 'last')

    train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                 batch_size=model.config.BATCH_SIZE,
                                 augment = False)    
    model.config.display()

    return [model, dataset_train, train_generator, config]
    
    
    
 
def prep_newshapes_dev(init_with = "last", FCN_layers= False):

    MODEL_DIR = os.path.join(MODEL_PATH, "mrcnn_logs")

    # Build configuration object -----------------------------------------------
    config = new_shapes.NewShapesConfig()
    config.BATCH_SIZE      = 5                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU  = 5                  # Must match BATCH_SIZE
    config.STEPS_PER_EPOCH = 4
    config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]

    # Build shape dataset        -----------------------------------------------
    # Training dataset
    # generate 500 shapes 
    dataset_train = new_shapes.NewShapesDataset()
    dataset_train.load_shapes(3000, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()

    # Validation dataset
    dataset_val = new_shapes.NewShapesDataset()
    dataset_val.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()

    try :
        del model, train_generator, val_generator, mm
        gc.collect()
    except: 
        pass
    KB.clear_session()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR,FCN_layers = FCN_layers)

    print('MODEL_PATH        : ', MODEL_PATH)
    print('COCO_MODEL_PATH   : ', COCO_MODEL_PATH)
    print('RESNET_MODEL_PATH : ', RESNET_MODEL_PATH)
    print('MODEL_DIR         : ', MODEL_DIR)
    print('Last Saved Model  : ', model.find_last())

    load_model(model, init_with = 'last')

    train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                 batch_size=model.config.BATCH_SIZE,
                                 augment = False)    
    config.display()     
    return [model, dataset_train, train_generator, config]
    
        
def load_model(model, init_with = None):
    '''
    methods to load weights
    1 - load a specific file
    2 - find a last checkpoint in a specific folder 
    3 - use init_with keyword 
    '''    
    # Which weights to start with?
    print('-----------------------------------------------')
    print('Load model with init parm: ', init_with)
    print('-----------------------------------------------')

    print(model.find_last())

   
    print('load model with <init_with> = ', init_with)
    #model.keras_model.summary(line_length = 120) 
    # model.compile_only(learning_rate=config.LEARNING_RATE, layers='heads')
    # KB.set_learning_phase(1)

    ## 1- look for a specific weights file 
    ## Load trained weights (fill in path to trained weights here)
    # model_path  = 'E:\\Models\\mrcnn_logs\\shapes20180428T1819\\mask_rcnn_shapes_5784.h5'
    # print(' model_path : ', model_path )

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
    # init_with = "last"  # imagenet, coco, or last

    if init_with == "imagenet":
    #     loc=model.load_weights(model.get_imagenet_weights(), by_name=True)
        loc=model.load_weights(RESNET_MODEL_PATH, by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        
        # See README for instructions to download the COCO weights
        loc=model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        loc= model.load_weights(model.find_last()[1], by_name=True)
    else:
        assert init_with != "", "Provide path to trained weights"
        print("Loading weights from ", init_with)
        loc = model.load_weights(model_path, by_name=True)    

        
    print('Load weights complete', loc)    