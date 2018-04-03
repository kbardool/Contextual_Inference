"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import glob
import random
import math
import datetime
import itertools
import json
import re
import logging
from collections import OrderedDict
import numpy as np
# import scipy.misc
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
from keras.regularizers     import l2
# import keras.initializers as KI
# import keras.engine as KE

sys.path.append('..')

from mrcnn.BilinearUpSampling import BilinearUpSampling2D

# import mrcnn.utils as utils
# from   mrcnn.datagen import data_generator
# import mrcnn.loss  as loss

# Requires TensorFlow 1.3+ and Keras 2.0.8+.


###############################################################
# Fully Convolutional Network 
###############################################################

# def fcn_graph(feature_map, anchors_per_location, anchor_stride):

def fcn_Vgg16_32s(feature_map, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=0):
    '''Builds the computation graph of Region Proposal Network.

    feature_map:            Contextual Tensor [batch, num_classes, width, depth]

    Returns:
        rpn_logits:     [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs:      [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox:       [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                        applied to anchors.
    '''
    print(' feature map shape is ', feature_map.shape)
    # TODO: Assert proper shape of input [batch_size, width, height, num_classes]
    
    # TODO: check if stride of 2 causes alignment issues if the featuremap is not even.
    
    # if batch_shape:
        # img_input = Input(batch_shape=batch_shape)
        # image_size = batch_shape[1:3]
    # else:
        # img_input = Input(shape=input_shape)
        # image_size = input_shape[0:2]
    
    # Block 1
    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(feature_map)
    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = KL.Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = KL.Dropout(0.5)(x)
    x = KL.Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = KL.Dropout(0.5)(x)
    #classifying layer
    x = KL.Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1),
                kernel_regularizer=l2(weight_decay))(x)

    output = BilinearUpSampling2D(size=(32, 32))(x)

    # model = Model(img_input, x)
    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    # weights_path = 'E:/Models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    # model.load_weights(weights_path, by_name=True)
    
    return output


    
 

def build_fcn_model(config):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.
    anchor_stride:          Controls the density of anchors. Typically 1 (anchors for
                            every pixel in the feature map), or 2 (every other pixel).

    anchors_per_location:   number of anchors per pixel in the feature map
    depth:                  Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_logits:         [batch, H, W, 2] Anchor classifier logits (before softmax)
    rpn_probs:          [batch, W, W, 2] Anchor classifier probabilities.
    rpn_bbox:           [batch, H, W, (dy, dx, log(dh), log(dw))] 
                        Deltas to be applied to anchors.
    """
    print('>>> FCN Layer ')
    height, width = config.FCN_INPUT_SHAPE[0:2]
    depth         = config.NUM_CLASSES
    print(' height :', height, 'width :', width, 'depth :' , depth)
    input_feature_map = KL.Input(shape=[ width, height, depth], name="input_fcn_feature_map")
    
    outputs = fcn_Vgg16_32s(input_feature_map, classes=config.NUM_CLASSES)
    
    return KM.Model([input_feature_map], outputs, name="fcn_model")

