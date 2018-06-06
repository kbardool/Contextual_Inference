"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
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

def normalize(x):
    #--------------------------------------------------------------------------------------------
    # this normalization moves values to [0  +1] 
    #--------------------------------------------------------------------------------------------    

    # x   = x / tf.reduce_max(x, axis=[1,2], keepdims = True)
    # x   = tf.where(tf.is_nan(x),  tf.zeros_like(x), x, name = 'fcn_heatmap_norm')

    #--------------------------------------------------------------------------------------------
    # this normalization moves values to [-1 +1] 
    #--------------------------------------------------------------------------------------------    
    reduce_max = tf.reduce_max(x, axis = [1,2], keepdims=True)
    reduce_min = tf.reduce_min(x, axis = [1,2], keepdims=True)  ## epsilon    = tf.ones_like(reduce_max) * 1e-7
    print('     size of reduce max is ', reduce_max.shape)
    y  = tf.divide( (x- reduce_min) , (reduce_max - reduce_min), name='fcn_heatmap_norm')
    print('     size of y is : ', y.shape)




    return y

    
###############################################################
# Fully Convolutional Network Layer 
###############################################################

# def fcn_layer(context_tensor, num_classes,weight_decay=0., batch_momentum=0.9):
def fcn_graph( feature_map , config , weight_decay=0.01, batch_momentum=0.9):
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
    
    ## , kernel_regularizer=l2(weight_decay)
    
    # Block 1    data_format='channels_last',
    
    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='fcn_block1_conv1')(feature_map)
    print('   FCN Block 11 shape is : ' ,x.get_shape())
    
    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='fcn_block1_conv2')(x)
    print('   FCN Block 12 shape is : ' ,x.get_shape())         
    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='fcn_block1_pool')(x)
    print('   FCN Block 13 shape is : ' ,x.get_shape())
    x0 = x
    
    # Block 2
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='fcn_block2_conv1')(x)
    print('   FCN Block 21 shape is : ' , x.get_shape())
    
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='fcn_block2_conv2')(x)
    print('   FCN Block 22 shape is : ' ,x.get_shape())    
    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='fcn_block2_pool')(x)
    print('   FCN Block 23 (Max pooling) shape is : ' ,x.get_shape())    
    x1 = x
    
    # Block 3
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='fcn_block3_conv1')(x)
    print('   FCN Block 31 shape is : ' ,x.get_shape())            
    
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='fcn_block3_conv2')(x)
    print('   FCN Block 32 shape is : ' ,x.get_shape())    
    
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='fcn_block3_conv3')(x)
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
    x = KL.Conv2D(FC_SIZE, (7, 7), activation='relu', padding='same', name='fcn_fc1')(x)
    print('   FCN fully connected 1 (fcn_fc1) shape is : ' ,x.get_shape())        
    x = KL.Dropout(0.5)(x)
    x = KL.Conv2D(FC_SIZE, (1, 1), activation='relu', padding='same', name='fcn_fc2')(x)
    print('   FCN fully connected 2 (fcn_fc2) shape is : ' ,x.get_shape())        
    x = KL.Dropout(0.5)(x)
    
    #classifying layer
    x = KL.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal', 
                  activation='linear', padding='valid', strides=(1, 1),
                  name='fcn_classify')(x)

    
    print('   FCN final conv2d (fcn_classify) shape is : ' , x.get_shape(),' keras_tensor ', KB.is_keras_tensor(x))                      
    
    fcn_classify_shape = KB.int_shape(x)
    h_factor = height / fcn_classify_shape[1]
    w_factor = height / fcn_classify_shape[2]
    print('   h_factor : ', h_factor, 'w_factor : ', w_factor)
    
    x = BilinearUpSampling2D(size=(h_factor, w_factor), name='fcn_bilinear')(x)
    print('   FCN Bilinear upsmapling layer  shape is : ' , x.get_shape(), ' Keras tensor ', KB.is_keras_tensor(x) )
    
    
    ##---------------------------------------------------------------------------------------------
    ## heatmap L2 normalization
    ## Normalization using the  `gauss_sum` (batchsize , num_classes, height, width) 
    ## 17-05-2018 (New method, replace dthe previous method that usedthe transposed gauss sum
    ## 17-05-2018 Replaced with normalization across the CLASS axis 
    ##                         normalize along the CLASS axis 
    ##---------------------------------------------------------------------------------------------
    fcn_hm = KL.Lambda(lambda z: tf.identity(z, name="fcn_heatmap"), name="fcn_heatmap") (x)

    print()
    print('\n    L2 normalization ------------------------------------------------------')   
    fcn_hm_L2norm = KL.Lambda(lambda z: tf.nn.l2_normalize(z, axis = 3, name = 'fcn_heatmap_L2norm'),\
                        name = 'fcn_heatmap_L2norm')(x)

    print('\n    normalization ------------------------------------------------------')   
    fcn_hm_norm   = KL.Lambda(normalize, name="fcn_heatmap_norm") (x)

    print('    fcn_heatmap       : ', fcn_hm.shape        ,' Keras tensor ', KB.is_keras_tensor(fcn_hm) )
    print('    fcn_heatmap_norm  : ', fcn_hm_norm.shape   ,' Keras tensor ', KB.is_keras_tensor(fcn_hm_norm) )
    print('    fcn_heatmap_L2norm: ', fcn_hm_L2norm.shape ,' Keras tensor ', KB.is_keras_tensor(fcn_hm_L2norm) )

    return fcn_hm_norm, fcn_hm, fcn_hm_L2norm

