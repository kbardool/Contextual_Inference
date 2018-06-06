"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import argparse

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
sys.path.append('../')

import mrcnn.utils   as utils
import mrcnn.model   as modellib
import mrcnn.dataset as dataset
from   mrcnn.config import Config
from   mrcnn.coco import CocoDataset, CocoConfig, evaluate_coco, build_coco_results


import pprint as pp
pp = pp.PrettyPrinter(indent=4)


##------------------------------------------------------------------------------------
## setup project directories
##------------------------------------------------------------------------------------
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


print(COCO_MODEL_PATH)
############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse
    #----------------------------------------------------------------------------
    ## Parse command line arguments
    #----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')
    # parser.add_argument("command",
                        # metavar="<command>",
                        # help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=False,
                        default=COCO_DATASET_PATH,                    
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=False,
                        default=COCO_MODEL_PATH,                    
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco', 'last','imagenet' ")
                        
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
                        
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
                        
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (defaults=500)')
                        
    args = parser.parse_args()
    args.command = 'train'
    # args = parser.parse_args("train --dataset E:\MLDatasets\coco2014 --model mask_rcnn_coco.h5 --limit 10".split())
    # args = parser.parse_args("train  --limit 10".split())
    # pp.pprint(options)
    pp.pprint(args)
    print(" Model     (COCO_MODEL_PATH)  : ", args.model, '  ', COCO_MODEL_PATH)
    print(" Dataset   (COCO_DATASET_PATH): ", args.dataset, '  ',COCO_DATASET_PATH)
    print(" Ckpt/Logs (DEFAULT_LOGS_DIR) : ", args.logs, '  ', DEFAULT_LOGS_DIR)
    print(" Limit:   ", args.limit)



    #------------------------------------------------------------------------------------
    ## setup tf session and debugging 
    #------------------------------------------------------------------------------------
    # import tensorflow as tf
    # import keras.backend as KB 
    # kears.backend.tensorflow_backend import set_session
    # keras_backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))
    #
    #     if 'tensorflow' == KB.backend():
    #         from tensorflow.python import debug as tf_debug
    #         config = tf.ConfigProto(
    #                 device_count = {'GPU': 0}
    #             )
    #         tf_sess = tf.Session(config=config)    
    #         tf_sess = tf_debug.LocalCLIDebugWrapperSession(tf_sess)
    #         KB.set_session(tf_sess)
    #------------------------------------------------------------------------------------
    # force no GPU usage
    #------------------------------------------------------------------------------------
    # if 'tensorflow' == KB.backend():
    #     config = tf.ConfigProto(
    #             device_count = {'GPU': 0}
    #         )
    #     tf_sess = tf.Session(config=config)    
    #     KB.set_session(tf_sess)
    #------------------------------------------------------------------------------------
    # limit GPU usage
    #------------------------------------------------------------------------------------
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

    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.55
    # set_session(tf.Session(config=tf_config))

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.55)
    # set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))    

    #----------------------------------------------------------------------------
    ## Configurations
    #----------------------------------------------------------------------------
    if args.command == "train":
        config = CocoConfig()
        config.BATCH_SIZE      = 1                  # Batch size is 2 (# GPUs * images/GPU).
        config.IMAGES_PER_GPU  = 1                  # Must match BATCH_SIZE
        config.STEPS_PER_EPOCH = int(args.steps_per_epoch)
        config.LEARNING_RATE   = float(args.lr)
        config.EPOCHS_TO_RUN   = int(args.epochs)
        # config.IMAGE_MAX_DIM   = 600
        # config.IMAGE_MIN_DIM   = 480
        
        # config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]
        # config.LAST_EPOCH_RAN  = int(args.last_epoch)
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    
    config.display()
    
    
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.55
    # set_session(tf.Session(config=tf_config))
    
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.55)
    # set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    
    #----------------------------------------------------------------------------
    ## Create model
    #----------------------------------------------------------------------------
    # Create model
    if args.command == "train":
        print('setup model for Training ')
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        print('setup model for Inference ')
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    #----------------------------------------------------------------------------
    ## Select weights file to load
    #----------------------------------------------------------------------------
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    #----------------------------------------------------------------------------
    ## Load weights
    #----------------------------------------------------------------------------
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    #----------------------------------------------------------------------------
    ## Train or evaluate
    #----------------------------------------------------------------------------
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train")
        dataset_train.load_coco(args.dataset, "val35k")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_coco(args.dataset, "minival")
        dataset_val.prepare()

        print('\n Outputs: ') 
        pp.pprint(model.keras_model.outputs)

        trainable = model.get_trainable_layers()
        for i in trainable:
            print(' Layer:', i.name)
        
        config.display()
         
        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=config.EPOCHS_TO_RUN,
                    layers='fcn')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')
    #-----------------------------------------------------
    # Evaluate 
    #-----------------------------------------------------
    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        coco = dataset_val.load_coco(args.dataset, "minival", return_coco=True)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
