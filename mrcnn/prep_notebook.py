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
import mrcnn.new_shapes as new_shapes
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

##------------------------------------------------------------------------------------    
## New Shapes TESTING
##------------------------------------------------------------------------------------    
def prep_newshapes_test(init_with = 'last', FCN_layers = False, batch_sz = 5, epoch_steps = 4,folder_name= "mrcnn_newshape_test_logs"):

    MODEL_DIR = os.path.join(MODEL_PATH, folder_name)

    # Build configuration object -----------------------------------------------
    config = new_shapes.NewShapesConfig()
    config.BATCH_SIZE      = batch_sz                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU  = batch_sz                  # Must match BATCH_SIZE
    config.STEPS_PER_EPOCH = epoch_steps
    config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]
    config.DETECTION_MIN_CONFIDENCE = 0.1
    # Build shape dataset        -----------------------------------------------
    # Training dataset
    dataset_test = new_shapes.NewShapesDataset(config)
    dataset_test.load_shapes(3000)
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
    # exclude_layers = \
           # ['fcn_block1_conv1' 
           # ,'fcn_block1_conv2' 
           # ,'fcn_block1_pool' 
           # ,'fcn_block2_conv1'
           # ,'fcn_block2_conv2' 
           # ,'fcn_block2_pool'  
           # ,'fcn_block3_conv1' 
           # ,'fcn_block3_conv2' 
           # ,'fcn_block3_conv3' 
           # ,'fcn_block3_pool'  
           # ,'fcn_block4_conv1' 
           # ,'fcn_block4_conv2' 
           # ,'fcn_block4_conv3' 
           # ,'fcn_block4_pool'  
           # ,'fcn_block5_conv1' 
           # ,'fcn_block5_conv2' 
           # ,'fcn_block5_conv3' 
           # ,'fcn_block5_pool'  
           # ,'fcn_fc1'          
           # ,'dropout_1'        
           # ,'fcn_fc2'          
           # ,'dropout_2'        
           # ,'fcn_classify'     
           # ,'fcn_bilinear'     
           # ,'fcn_heatmap_norm' 
           # ,'fcn_scoring'      
           # ,'fcn_heatmap'      
           # ,'fcn_norm_loss']
    
    # load_model(model, init_with = init_with, exclude = exclude_layers)
    model.load_model_weights(init_with = init_with) 

    # print('=====================================')
    # print(" Load second weight file ?? ")
    # model.keras_model.load_weights('E:/Models/vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name= True )
    
    test_generator = data_generator(dataset_test, model.config, shuffle=True,
                                     batch_size=model.config.BATCH_SIZE,
                                     augment = False)
    model.config.display()     
    return [model, dataset_test, test_generator, config]                                 
    
    
    
##------------------------------------------------------------------------------------    
## New Shapes TRAINING 
##------------------------------------------------------------------------------------            
def prep_newshapes_train2(init_with = "last",  config=None):

    import mrcnn.new_shapes as new_shapes
    config.CHECKPOINT_FOLDER = os.path.join(MODEL_PATH, config.CHECKPOINT_FOLDER)

    # Build shape dataset        -----------------------------------------------
    # Training dataset
    dataset_train = new_shapes.NewShapesDataset(config)
    dataset_train.load_shapes(config.TRAINING_IMAGES) 
    dataset_train.prepare()

    # Validation dataset
    dataset_val = new_shapes.NewShapesDataset(config)
    dataset_val.load_shapes(config.VALIDATION_IMAGES)
    dataset_val.prepare()

    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=config.CHECKPOINT_FOLDER, FCN_layers = config.FCN_LAYERS)

    print('MODEL_PATH        : ', MODEL_PATH)
    print('COCO_MODEL_PATH   : ', COCO_MODEL_PATH)
    print('RESNET_MODEL_PATH : ', RESNET_MODEL_PATH)
    print('CHECKPOINT_DIR    : ', config.CHECKPOINT_FOLDER)
    print('Last Saved Model  : ', model.find_last())
    # exclude_layers = \
           # ['fcn_block1_conv1' 
           # ,'fcn_block1_conv2' 
           # ,'fcn_block1_pool' 
           # ,'fcn_block2_conv1'
           # ,'fcn_block2_conv2' 
           # ,'fcn_block2_pool'  
           # ,'fcn_block3_conv1' 
           # ,'fcn_block3_conv2' 
           # ,'fcn_block3_conv3' 
           # ,'fcn_block3_pool'  
           # ,'fcn_block4_conv1' 
           # ,'fcn_block4_conv2' 
           # ,'fcn_block4_conv3' 
           # ,'fcn_block4_pool'  
           # ,'fcn_block5_conv1' 
           # ,'fcn_block5_conv2' 
           # ,'fcn_block5_conv3' 
           # ,'fcn_block5_pool'  
           # ,'fcn_fc1'          
           # ,'dropout_1'        
           # ,'fcn_fc2'          
           # ,'dropout_2'        
           # ,'fcn_classify'     
           # ,'fcn_bilinear'     
           # ,'fcn_heatmap_norm' 
           # ,'fcn_scoring'      
           # ,'fcn_heatmap'      
           # ,'fcn_norm_loss']
    # load_model(model, init_with = 'last', exclude = exclude_layers)
    model.load_model_weights(init_with = init_with)
    
    # print('=====================================')
    # print(" Load second weight file ?? ")
    # model.keras_model.load_weights('E:/Models/vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name= True)
    
    
    
    train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                 batch_size=model.config.BATCH_SIZE,
                                 augment = False)   


    val_generator = data_generator(dataset_val, model.config, shuffle=True, 
                                    batch_size=model.config.BATCH_SIZE,
                                    augment=False)                                           
    config.display()     
    return [model, dataset_train, dataset_val, train_generator, val_generator, config]

    
##------------------------------------------------------------------------------------    
## New Shapes TRAINING 
##------------------------------------------------------------------------------------            
def prep_newshapes_train(init_with = "last", FCN_layers= False, batch_sz =5, epoch_steps = 4, folder_name= None):

    MODEL_DIR = os.path.join(MODEL_PATH, folder_name)

    # Build configuration object -----------------------------------------------
    config = new_shapes.NewShapesConfig()
    config.BATCH_SIZE      = batch_sz                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU  = batch_sz                  # Must match BATCH_SIZE
    config.STEPS_PER_EPOCH = epoch_steps
    config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]

    # Build shape dataset        -----------------------------------------------
    # Training dataset
    dataset_train = new_shapes.NewShapesDataset(config)
    dataset_train.load_shapes(10000) 
    dataset_train.prepare()

    # Validation dataset
    dataset_val = new_shapes.NewShapesDataset(config)
    dataset_val.load_shapes(2500)
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
    # exclude_layers = \
           # ['fcn_block1_conv1' 
           # ,'fcn_block1_conv2' 
           # ,'fcn_block1_pool' 
           # ,'fcn_block2_conv1'
           # ,'fcn_block2_conv2' 
           # ,'fcn_block2_pool'  
           # ,'fcn_block3_conv1' 
           # ,'fcn_block3_conv2' 
           # ,'fcn_block3_conv3' 
           # ,'fcn_block3_pool'  
           # ,'fcn_block4_conv1' 
           # ,'fcn_block4_conv2' 
           # ,'fcn_block4_conv3' 
           # ,'fcn_block4_pool'  
           # ,'fcn_block5_conv1' 
           # ,'fcn_block5_conv2' 
           # ,'fcn_block5_conv3' 
           # ,'fcn_block5_pool'  
           # ,'fcn_fc1'          
           # ,'dropout_1'        
           # ,'fcn_fc2'          
           # ,'dropout_2'        
           # ,'fcn_classify'     
           # ,'fcn_bilinear'     
           # ,'fcn_heatmap_norm' 
           # ,'fcn_scoring'      
           # ,'fcn_heatmap'      
           # ,'fcn_norm_loss']
    # load_model(model, init_with = 'last', exclude = exclude_layers)
    model.load_model_weights(init_with = init_with)
    
    # print('=====================================')
    # print(" Load second weight file ?? ")
    # model.keras_model.load_weights('E:/Models/vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name= True)
    
    
    
    train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                 batch_size=model.config.BATCH_SIZE,
                                 augment = False)   


    val_generator = data_generator(dataset_val, model.config, shuffle=True, 
                                    batch_size=model.config.BATCH_SIZE,
                                    augment=False)                                           
    config.display()     
    return [model, dataset_train, dataset_val, train_generator, val_generator, config]

    

    
##------------------------------------------------------------------------------------    
## Old Shapes TRAINING
##------------------------------------------------------------------------------------   
def prep_oldshapes_train(init_with = None, FCN_layers = False, batch_sz = 5, epoch_steps = 4, folder_name= "mrcnn_oldshape_training_logs"):
    import mrcnn.shapes    as shapes
    MODEL_DIR = os.path.join(MODEL_PATH, folder_name)

    # Build configuration object -----------------------------------------------
    config = shapes.ShapesConfig()
    config.BATCH_SIZE      = batch_sz                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU  = batch_sz                  # Must match BATCH_SIZE
    config.STEPS_PER_EPOCH = epoch_steps
    config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]

    # Build shape dataset        -----------------------------------------------
    dataset_train = shapes.ShapesDataset(config)
    dataset_train.load_shapes(3000) 
    dataset_train.prepare()

    # Validation dataset
    dataset_val  = shapes.ShapesDataset(config)
    dataset_val.load_shapes(500)
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

    model.load_model_weights(init_with = init_with)

    train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                     batch_size=model.config.BATCH_SIZE,
                                     augment = False)
    val_generator = data_generator(dataset_val, model.config, shuffle=True, 
                                    batch_size=model.config.BATCH_SIZE,
                                    augment=False)                                 
    model.config.display()     
    return [model, dataset_train, dataset_val, train_generator, val_generator, config]                                 

##------------------------------------------------------------------------------------    
## Old Shapes TESTING
##------------------------------------------------------------------------------------    
    
def prep_oldshapes_test(init_with = None, FCN_layers = False, batch_sz = 5, epoch_steps = 4, folder_name= "mrcnn_oldshape_test_logs"):
    import mrcnn.shapes as shapes
    MODEL_DIR = os.path.join(MODEL_PATH, folder_name)
    # MODEL_DIR = os.path.join(MODEL_PATH, "mrcnn_development_logs")

    # Build configuration object -----------------------------------------------
    config = shapes.ShapesConfig()
    config.BATCH_SIZE      = batch_sz                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU  = batch_sz                  # Must match BATCH_SIZE
    config.STEPS_PER_EPOCH = epoch_steps
    config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]

    # Build shape dataset        -----------------------------------------------
    dataset_test = shapes.ShapesDataset(config)
    dataset_test.load_shapes(500) 
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

    model.load_model_weights(init_with = init_with)

    test_generator = data_generator(dataset_test, model.config, shuffle=True,
                                     batch_size=model.config.BATCH_SIZE,
                                     augment = False)
    model.config.display()     
    return [model, dataset_test, test_generator, config]                                 


"""
##------------------------------------------------------------------------------------    
## Old Shapes DEVELOPMENT
##------------------------------------------------------------------------------------        
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




    
##------------------------------------------------------------------------------------    
## New Shapes DEVELOPMENT
##------------------------------------------------------------------------------------            
def prep_newshapes_dev(init_with = "last", FCN_layers= False, batch_sz = 5):
    import mrcnn.new_shapes as new_shapes
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
    
"""
