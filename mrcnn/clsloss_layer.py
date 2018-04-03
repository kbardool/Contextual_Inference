"""
Mask R-CNN
Dataset functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy as np
import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE

import mrcnn.utils as utils


def mrcnn_class_loss_graph_2(target_class_ids, pred_class_logits, active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids:       [batch, num_rois]. Integer class IDs. Uses zero
                            padding to fill in the array.
    pred_class_logits:      [batch, num_rois, num_classes]
    active_class_ids:       [batch, num_classes]. Has a value of 1 for
                            classes that are in the dataset of the image, and 0
                            for classes that are not in the dataset.
    """
    print('>>>  mrcnn_class_loss_graph_2()')
    print('     target_class_ids ', type(target_class_ids) , target_class_ids.shape  )
    print('     pred_class_logits', type(pred_class_logits), pred_class_logits.shape )
    print('     active_class_ids ', type(active_class_ids) , active_class_ids.shape  )
    sess = tf.InteractiveSession()
    target_class_ids = tf.cast(target_class_ids, 'int64')
    # target_class_ids = target_class_ids.astype(np.int64)
    print('target_class_ids \n')
    print(target_class_ids.eval())
    
    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    print('pred_class_ids  (argmaxs)  \n', type(pred_class_ids))
    print(pred_class_ids.eval())
    
    # print('pred_class_logits :', type(pred_class_logits))
    # print(pred_class_logits)

    print('active_class_ids \n', active_class_ids)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    print('pred_active ', pred_active.shape)
    print(pred_active.eval())
    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active
    # print('loss after multuplication by pred_active')
    # print(loss.eval())

    # Compute loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    print(' tf.reduce_sum(loss) \n',tf.reduce_sum(loss).eval())
    print()
    print(' tf.reduce_sum(pred_active) \n',tf.reduce_sum(pred_active).eval())
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    print('    mrcnn_class_loss_graph_2 : Loss is : ',loss.eval())
    
    np_loss = loss.eval()
    print('    np_loss ---> ',np_loss)
    return np_loss
    
    
class CLSLossLayer(KE.Layer):
    """

    Returns:
    -------

    """

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        print('>>> CLSLoss Layer : initialization')
        self.config = config

        
    def call(self, inputs):
    
        print('>>> CLS Loss Layer : call')
        
        print('    target_class_ids   .shape/type  :',  inputs[0].shape) # , type(inputs[0]))
        print('    mrcnn_class_logits .shape/type  :',  inputs[1].shape) # , type(inputs[1])) 
        print('    activate_class_ids .shape/type  :',  inputs[2].shape) # , type(inputs[2])) 
        
        target_class_ids   =  inputs[0]
        mrcnn_class_logits =  inputs[1]
        activate_class_ids =  inputs[2]
        
        def wrapper(target_class_ids, mrcnn_class_logits, activate_class_ids):
            print('>>> CLS Loss Layer Wrapper: Begin')
            loss = mrcnn_class_loss_graph_2(target_class_ids, mrcnn_class_logits, activate_class_ids)
            print('    Loss: ', loss.shape, '   ', loss)    
            print('>>> CLS Loss Layer Wrapper: End', type(loss))
            return [loss]

        # Return wrapped function
        # print('>>> CLS Loss Layer : call end  ')

        return tf.py_func(wrapper, inputs, [tf.float32])

        
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        return [(1) ]

    