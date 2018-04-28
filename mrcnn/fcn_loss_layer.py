"""
Mask R-CNN
Dataset functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy              as np
import tensorflow         as tf
import keras.backend      as KB
import keras.layers       as KL
import keras.initializers as KI
import keras.engine       as KE
import mrcnn.utils        as utils
from  mrcnn.loss import smooth_l1_loss 
import pprint
pp = pprint.PrettyPrinter(indent=2, width=100)

##-----------------------------------------------------------------------
##  FCN loss
##-----------------------------------------------------------------------    
def fcn_loss_graph(target_masks, pred_masks):
# def fcn_loss_graph(input):
    # target_masks, pred_masks = input
    """Mask binary cross-entropy loss for the masks head.

    target_masks:       [batch, height, width, num_classes].
    
    pred_masks:         [batch, height, width, num_classes] float32 tensor
    """
    # Reshape for simplicity. Merge first two dimensions into one.

    print('\n    fcn_loss_graph ' )
    print('    target_masks     shape :', target_masks.get_shape())
    print('    pred_masks       shape :', pred_masks.get_shape())    
    
    mask_shape       = tf.shape(target_masks)
    print('    mask_shape       shape :', mask_shape.shape)    
    
    target_masks     = KB.reshape(target_masks, (-1, mask_shape[1], mask_shape[2]))
    print('    target_masks     shape :', target_masks.shape)        
    
    pred_shape       = tf.shape(pred_masks)
    print('    pred_shape       shape :', pred_shape.shape)        
    
    pred_masks       = KB.reshape(pred_masks, (-1, pred_shape[1], pred_shape[2]))
    print('    pred_masks       shape :', pred_masks.get_shape())        

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    # Smooth-L1 Loss
    loss        = KB.switch(tf.size(target_masks) > 0,
                    smooth_l1_loss(y_true=target_masks, y_pred=pred_masks),
                    tf.constant(0.0))
    loss        = KB.mean(loss)
    loss        = KB.reshape(loss, [1, 1])
    print('    loss type is :', type(loss))
    return loss

##-----------------------------------------------------------------------
##  FCN loss for L2 Normalized graph
##-----------------------------------------------------------------------    
    
def fcn_norm_loss_graph(target_masks, pred_masks):
    '''
    Mask binary cross-entropy loss for the masks head.
    target_masks:       [batch, height, width, num_classes].
    pred_masks:         [batch, height, width, num_classes] float32 tensor
    '''
    print(type(target_masks))
    pp.pprint(dir(target_masks))
    # Reshape for simplicity. Merge first two dimensions into one.
    print('\n    fcn_norm_loss_graph ' )
    print('    target_masks     shape :', target_masks.shape)
    print('    pred_masks       shape :', pred_masks.shape)    
    print('\n    L2 normalization ------------------------------------------------------')   
    output_shape=KB.int_shape(pred_masks)
    print(' output shape is :' , output_shape, '   ', pred_masks.get_shape(), pred_masks.shape, tf.shape(pred_masks))
    
    output_flatten = KB.reshape(pred_masks, (pred_masks.shape[0], -1, pred_masks.shape[-1]) )
    output_norm1   = KB.l2_normalize(output_flatten, axis = 1)    
    output_norm    = KB.reshape(output_norm1,  KB.shape(pred_masks) )
    
    print('   output_flatten    : ', KB.int_shape(output_flatten) , ' Keras tensor ', KB.is_keras_tensor(output_flatten) )
    print('   output_norm1      : ', KB.int_shape(output_norm1)   , ' Keras tensor ', KB.is_keras_tensor(output_norm1) )
    print('   output_norm final : ', KB.int_shape(output_norm)    , ' Keras tensor ', KB.is_keras_tensor(output_norm) )

    pred_masks1 = output_norm


    print('\n    L2 normalization ------------------------------------------------------')         
    
    gauss_flatten = KB.reshape(target_masks, (target_masks.shape[0], -1, target_masks.shape[-1]) )
    gauss_norm1   = KB.l2_normalize(gauss_flatten, axis = 1)
    gauss_norm    = KB.reshape(gauss_norm1, KB.shape(target_masks))
    
    print('    guass_flatten         : ', KB.int_shape(gauss_flatten), 'Keras tensor ', KB.is_keras_tensor(gauss_flatten) )
    print('    gauss_norm shape      : ', KB.int_shape(gauss_norm1)  , 'Keras tensor ', KB.is_keras_tensor(gauss_norm1) )
    print('    gauss_norm final shape: ', KB.int_shape(gauss_norm)   , 'Keras tensor ', KB.is_keras_tensor(gauss_norm) )
    print('    complete')

    target_masks1 = gauss_norm


    
    mask_shape       = tf.shape(target_masks1)
    print('    mask_shape       shape :', mask_shape.shape)    
    
    target_masks1     = KB.reshape(target_masks1, (-1, mask_shape[1], mask_shape[2]))
    print('    target_masks     shape :', target_masks1.shape)        
    
    pred_shape       = tf.shape(pred_masks1)
    print('    pred_shape       shape :', pred_shape.shape)        
    
    pred_masks1       = KB.reshape(pred_masks1, (-1, pred_shape[1], pred_shape[2]))
    print('    pred_masks       shape :', pred_masks1.get_shape())        

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    # Smooth-L1 Loss
    loss        = KB.switch(tf.size(target_masks1) > 0,
                    smooth_l1_loss(y_true=target_masks1, y_pred=pred_masks1),
                    tf.constant(0.0))
    loss        = KB.mean(loss)
    loss        = KB.reshape(loss, [1, 1])
    print('    loss type is :', type(loss))
    return loss
         
    
class FCNLossLayer(KE.Layer):
    """

    Returns:
    -------

    """

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        print('>>> FCN Loss Layer : initialization')
        self.config = config

        
    def call(self, inputs):
    
        print('\n    FCN Loss Layer : call')        
        print('    target_masks   .shape/type  :',  inputs[0].shape) # , type(inputs[0]))
        print('    pred_masks      shape/type  :',  inputs[1].shape) # , type(inputs[1])) 
        
        target_masks  =  inputs[0]
        pred_masks    =  inputs[1]
        loss           = KB.placeholder(shape=(1), dtype = 'float32', name = 'fcn_loss')
        norm_loss      = KB.placeholder(shape=(1), dtype = 'float32', name = 'fcn_norm_loss')
        loss      = fcn_loss_graph(target_masks, pred_masks)
        norm_loss = fcn_norm_loss_graph(target_masks, pred_masks)

        return [loss, norm_loss]
        
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        return [(1), (1)]

    