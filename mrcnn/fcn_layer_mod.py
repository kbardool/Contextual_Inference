"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
# import glob
# import random
# import math
# import datetime
# import itertools
# import json
# import re
# import logging
# from collections import OrderedDict
import numpy as np
# import scipy.misc
import tensorflow as tf
# import keras
import keras.backend as KB
import keras.layers as KL
import keras.models as KM
from   keras.regularizers     import l2
# import keras.initializers as KI
# import keras.engine as KE

sys.path.append('..')

from mrcnn.BilinearUpSampling import BilinearUpSampling2D
from   mrcnn.batchnorm_layer import BatchNorm
# import mrcnn.utils as utils
# from   mrcnn.datagen import data_generator
# import mrcnn.loss  as loss

# Requires TensorFlow 1.3+ and Keras 2.0.8+.

 
    
###############################################################
# Fully Convolutional Network Layer 
###############################################################

# def fcn_layer(context_tensor, num_classes,weight_decay=0., batch_momentum=0.9):
def fcn_graph_mod( feature_map , config , weight_decay=0., batch_momentum=0.9):
    '''Builds the computation graph of Region Proposal Network.

    feature_map:            Contextual Tensor [batch, num_classes, width, depth]

    Returns:


    '''
    print('\n>>> FCN Layer ')
    height, width = config.FCN_INPUT_SHAPE[0:2]
    num_classes   = config.NUM_CLASSES
    rois_per_class    = config.TRAIN_ROIS_PER_IMAGE
    
    print('     feature map shape is ', feature_map.shape)
    print('     height :', height, 'width :', width, 'classes :' , num_classes)
    print('     image_data_format: ', KB.image_data_format())
    print('     rois_per_class   : ', KB.image_data_format())
    feature_map_shape = (width, height, num_classes)
    
    # feature_map = KL.Input(shape= feature_map_shape, name="input_fcn_feature_map")
    # TODO: Assert proper shape of input [batch_size, width, height, num_classes]
    
    # TODO: check if stride of 2 causes alignment issues if the featuremap is not even.
    
    # if batch_shape:
        # img_input = Input(batch_shape=batch_shape)
        # image_size = batch_shape[1:3]
    # else:
        # img_input = Input(shape=input_shape)
        # image_size = input_shape[0:2]
    
    
    # Block 1    data_format='channels_last',
    x = KL.Conv2D(64, (3, 3), 
         activation='relu', padding='same', name='fcn_block1_conv1', 
         kernel_regularizer=l2(weight_decay))(feature_map)
    print('   FCN Block 11 shape is : ' ,x.get_shape())
    
    x  = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='fcn_block1_conv2', 
         kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 12 shape is : ' ,x.get_shape())         
    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='fcn_block1_pool')(x)
    print('   FCN Block 13 shape is : ' ,x.get_shape())
    x0 = x
    
    # Block 2
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', 
        name='fcn_block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 21 shape is : ' , x.get_shape())
    
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', 
        name='fcn_block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 22 shape is : ' ,x.get_shape())    
    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='fcn_block2_pool')(x)
    print('   FCN Block 23 (Max pooling) shape is : ' ,x.get_shape())    
    x1 = x
    
    # Block 3
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', 
        name='fcn_block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 31 shape is : ' ,x.get_shape())            
    
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', 
        name='fcn_block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 32 shape is : ' ,x.get_shape())    
    
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', 
        name='fcn_block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 33 shape is : ' ,x.get_shape())            
    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='fcn_block3_pool')(x)
    print('   FCN Block 34 (Max pooling) shape is : ' ,x.get_shape())    
    
    # Block 4
    # x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='fcn_block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    # x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='fcn_block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    # x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='fcn_block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    # x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='fcn_block4_pool')(x)

    # Block 5
    # x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='fcn_block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    # x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='fcn_block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    # x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='fcn_block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    # x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='fcn_block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    # changed from 4096 to 2048 - reduction of weights from 42,752,644 to                       
    # changed ftom 2048 to 1024 - 11-05-2018
    
    FC_SIZE = 1024 
    x = KL.Conv2D(FC_SIZE, (7, 7), activation='relu', padding='same', name='fcn_fc1', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN fully connected 1 (fcn_fc1) shape is : ' ,x.get_shape())        
    x = KL.Dropout(0.5)(x)
    x = KL.Conv2D(FC_SIZE, (1, 1), activation='relu', padding='same', name='fcn_fc2', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN fully connected 2 (fcn_fc2) shape is : ' ,x.get_shape())        
    x = KL.Dropout(0.5)(x)
    
    #classifying layer
    x = KL.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal', 
                  activation='linear', padding='valid', strides=(1, 1),
                  kernel_regularizer=l2(weight_decay), name='fcn_classify')(x)
    fcn_classify_shape = KB.int_shape(x)
    
    print('   FCN final conv2d (fcn_classify) shape is : ' , fcn_classify_shape )                      
    x2 = x
    h_factor = height / fcn_classify_shape[1]
    w_factor = height / fcn_classify_shape[2]
    print('   h_factor : ', h_factor, 'w_factor : ', w_factor)
    
    fcn_heatmap = BilinearUpSampling2D(size=(h_factor, w_factor), name='fcn_heatmap')(x)
    # print('   FCN Bilinear upsmapling layer  shape is : ' , x.get_shape(), 'Keras tensor ', KB.is_keras_tensor(x) )
    
    # heatmap  = KL.Lambda(lambda x: tf.transpose(x,[0,3,1,2]), name="fcn_heatmap") (x)
    # heatmap = KB.identity(tf.transpose( x,[0,3,1,2]), name = 'fcn_heatmap')
    print('   FCN heatmap (fcn_heatmap) shape is : ' , fcn_heatmap.get_shape(), 'Keras tensor ', KB.is_keras_tensor(fcn_heatmap) )
    print()
##---------------------------------------------------------------------------------------------------------------------------
## To normalize or to not normalize....?
##---------------------------------------------------------------------------------------------------------------------------
    
    # print('\n    L2 normalization ------------------------------------------------------')   
    # heatmap_shape=KB.int_shape(heatmap)
    # print(' heatmap shape is :' , heatmap_shape, '   ', heatmap.get_shape(), heatmap.shape, tf.shape(heatmap).eval())
    
    # heatmap_flat = tf.reshape(heatmap, [heatmap.shape[0], -1, heatmap.shape[-1]] )
    # print('   heatmap_flatten shape is : ', KB.int_shape(heatmap_flat), ' Keras tensor ', KB.is_keras_tensor(heatmap_flat) )
    
    # heatmap_norm = KB.l2_normalize(heatmap_flat, axis = 1)    
    # print('   heatmap_norm    shape is : ', KB.int_shape(heatmap_norm), ' Keras tensor ', KB.is_keras_tensor(heatmap_norm) )

    # heatmap_norm = KB.reshape(heatmap,  heatmap.shape )
    # print('   heatmap_norm final   shape is : ', KB.int_shape(heatmap_norm), ' Keras tensor ', KB.is_keras_tensor(heatmap_norm) )
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    pool_size = 7
    # x = KL.TimeDistributed(KL.Conv2D(1024, (pool_size, pool_size), padding="valid"), name="fcn_class_conv1")(heatmap)
    x = KL.Conv2D(128, (pool_size, pool_size), padding="valid", dilation_rate=2,  name="fcn_class_conv1")(fcn_heatmap)
    print('     fcn_class_conv1    output shape is : ' , x.get_shape())
    x = BatchNorm(name='fcn_class_bn1')(x)
    print('     fcn_class_bn1      output shape is : ' , x.get_shape())
    x  = KL.Activation('relu')(x)
    print('     fcn_class_relu1    output shape is : ' , x.get_shape())
    
    x = KL.Dropout(0.5)(x)
    x = KL.Conv2D( 64, (pool_size, pool_size), padding="valid", dilation_rate=2, name="fcn_class_conv2")(x)
    print('     fcn_class_conv2    output shape is : ' , x.get_shape())    
    x = BatchNorm(name='fcn_class_bn2')(x)
    print('     fcn_class_bn2      output shape is : ' , x.get_shape())
    shared = KL.Activation('relu')(x)
    print('     fcn_class_relu2    output shape is : ' , shared.get_shape())
    
    x = KL.Dropout(0.5)(x)
    x = KL.Conv2D( 32, (pool_size, pool_size), padding="valid", dilation_rate = 2, name="fcn_class_conv3")(x)
    print('     fcn_class_conv3    output shape is : ' , x.get_shape())    
    x = BatchNorm(name='fcn_class_bn3')(x)
    print('     fcn_class_bn3      output shape is : ' , x.get_shape())
    x = KL.Activation('relu')(x)
    print('     fcn_class_relu3    output shape is : ' , x.get_shape())

    shared = KL.Flatten()(x)
    # shared = KL.Lambda(lambda x: KB.squeeze(KB.squeeze(x, 3), 2), name="pool_squeeze")(x)
    
    print('     Input to Classifier / BBox heads : ' , shared.get_shape())

    # Classifier head
    fcn_class_logits = KL.Dense(num_classes * rois_per_class ,name='fcn_class_logits')(shared)
    print('     fcn_class_logits   output shape is : ' , fcn_class_logits.get_shape())    
    
    fcn_probs = KL.Activation("softmax", name="fcn_probs")(fcn_class_logits)
    print('     fcn_class_probs    output shape is : ' , fcn_probs.get_shape())    
    fcn_scores = KL.Reshape((num_classes, rois_per_class), name="fcn_scores")(fcn_probs)
    print('     fcn_scores         output shape is : ' , fcn_scores.get_shape())    

    # BBox head
    # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
    x = KL.Dense(num_classes * rois_per_class * 4, activation='linear',name='fcn_bbox_fc')(shared)
    print('     Dense layer        output shape is : ' , x.get_shape(), KB.int_shape(x))    
    # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
    s = KB.int_shape(x)
    
    fcn_bbox_deltas = KL.Reshape((num_classes, rois_per_class, 4), name="fcn_bbox_deltas")(x)
    print('     fcn_bbox_deltas    output shape is : ' , fcn_bbox_deltas.get_shape())    



    # return [x0, x1, x2,  output]
    return [fcn_heatmap, fcn_class_logits, fcn_scores, fcn_bbox_deltas]

