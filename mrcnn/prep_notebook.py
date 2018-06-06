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
from mrcnn.config      import Config
from mrcnn.dataset     import Dataset 
from mrcnn.utils       import stack_tensors, stack_tensors_3d, log
from mrcnn.datagen     import data_generator, load_image_gt
import platform

syst = platform.system()
if syst == 'Windows':
    # Root directory of the project
    print(' windows ' , syst)
    # WINDOWS MACHINE ------------------------------------------------------------------
    ROOT_DIR          = "E:\\"
    MODEL_PATH        = os.path.join(ROOT_DIR, "models")
    DATASET_PATH      = os.path.join(ROOT_DIR, 'MLDatasets')
    #### MODEL_DIR    = os.path.join(MODEL_PATH, "mrcnn_logs")
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
    #### MODEL_DIR    = os.path.join(MODEL_PATH, "mrcnn_development_logs")
    COCO_MODEL_PATH   = os.path.join(MODEL_PATH, "mask_rcnn_coco.h5")
    COCO_DATASET_PATH = os.path.join(DATASET_PATH,"coco2014")
    DEFAULT_LOGS_DIR  = os.path.join(MODEL_PATH, "mrcnn_coco_logs")
    RESNET_MODEL_PATH = os.path.join(MODEL_PATH, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
else :
    raise Error('unreconized system  '      )



print("Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))
import pprint
pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)

def build_config(batch_sz = 5, newshapes = False):
    if newshapes:
        import mrcnn.new_shapes as shapes
    else:
        import mrcnn.shapes    as shapes
        
    # Build configuration object -----------------------------------------------
    config = shapes.ShapesConfig()
    config.BATCH_SIZE      = batch_sz                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU  = batch_sz                  # Must match BATCH_SIZE
    config.STEPS_PER_EPOCH = 4
    config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]

    return config
    
def prep_oldshapes_dev(init_with = None, FCN_layers = False, batch_sz = 5):
    import mrcnn.shapes    as shapes
    MODEL_DIR = os.path.join(MODEL_PATH, "mrcnn_oldshape_dev_logs")

    config = build_config(batch_sz = batch_sz)

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

    load_model(model, init_with = init_with)

    train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                 batch_size=model.config.BATCH_SIZE,
                                 augment = False)    
    model.config.display()

    return [model, dataset_train, train_generator, config]
    

    
def prep_oldshapes_train(init_with = None, FCN_layers = False, batch_sz = 5):
    import mrcnn.shapes    as shapes
    MODEL_DIR = os.path.join(MODEL_PATH, "mrcnn_oldshape_train_logs")

    config = build_config(batch_sz = batch_sz)

    # Build shape dataset        -----------------------------------------------
    dataset_train = shapes.ShapesDataset()
    dataset_train.load_shapes(3000, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()

    # Validation dataset
    dataset_val  = shapes.ShapesDataset()
    dataset_val.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()
    
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

    load_model(model, init_with = init_with)

    train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                     batch_size=model.config.BATCH_SIZE,
                                     augment = False)
    val_generator = data_generator(dataset_val, model.config, shuffle=True, 
                                    batch_size=model.config.BATCH_SIZE,
                                    augment=False)                                 
    model.config.display()     
    return [model, dataset_train, dataset_val, train_generator, val_generator, config]                                 
    

def prep_oldshapes_test(init_with = None, FCN_layers = False, batch_sz = 5):
    import mrcnn.shapes as shapes
    MODEL_DIR = os.path.join(MODEL_PATH, "mrcnn_oldshape_test_logs")
    # MODEL_DIR = os.path.join(MODEL_PATH, "mrcnn_development_logs")

    config = build_config(batch_sz = batch_sz)

    # Build shape dataset        -----------------------------------------------
    dataset_test = shapes.ShapesDataset()
    dataset_test.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_test.prepare()

    # Recreate the model in inference mode
    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    model = modellib.MaskRCNN(mode="inference", 
                              config=config,
                              model_dir=MODEL_DIR, 
                              FCN_layers = FCN_layers )
        
    print(' COCO Model Path       : ', COCO_MODEL_PATH)
    print(' Checkpoint folder Path: ', MODEL_DIR)
    print(' Model Parent Path     : ', MODEL_PATH)
    print(' Resent Model Path     : ', RESNET_MODEL_PATH)

    load_model(model, init_with = init_with)

    test_generator = data_generator(dataset_test, model.config, shuffle=True,
                                     batch_size=model.config.BATCH_SIZE,
                                     augment = False)
    model.config.display()     
    return [model, dataset_test, test_generator, config]                                 
    
    
##------------------------------------------------------------------------------------    
## New Shapes 
##------------------------------------------------------------------------------------    
    
def prep_newshapes_dev(init_with = "last", FCN_layers= False, batch_sz = 5):
    import mrcnn.new_shapes as shapes
    MODEL_DIR = os.path.join(MODEL_PATH, "mrcnn_newshape_dev_logs")

    config = build_config(batch_sz = batch_sz, newshapes=True)

    # Build shape dataset        -----------------------------------------------
    # Training dataset 
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
    
        
def prep_newshapes_train(init_with = "last", FCN_layers= False, batch_sz =5):
    import mrcnn.new_shapes as shapes
    MODEL_DIR = os.path.join(MODEL_PATH, "mrcnn_newshape_training_logs")

    config = build_config(batch_sz = batch_sz, newshapes=True)

    # Build shape dataset        -----------------------------------------------
    # Training dataset
    dataset_train = new_shapes.NewShapesDataset()
    dataset_train.load_shapes(3000, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()

    # Validation dataset
    dataset_val = new_shapes.NewShapesDataset()
    dataset_val.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()

    try :
        del model
        print('delete model is successful')
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
    print(' Load model with init parm: ', init_with)
    print(' find last chkpt :', model.find_last())
    print('-----------------------------------------------')
   
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
        # Load the last model you trained and continue training, placing checkpouints in same folder
        loc= model.load_weights(model.find_last()[1], by_name=True)
    else:
        assert init_with != "", "Provide path to trained weights"
        print("Loading weights from ", init_with)
        loc = model.load_weights(init_with, by_name=True)    

        
    print('Load weights complete', loc)    