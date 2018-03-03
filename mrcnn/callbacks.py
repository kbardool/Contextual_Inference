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
import time
import json
import re
import logging
from collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM


def get_layer_output(input_layer, output_layer):
    _mrcnn_class = K.function([self.model.layers[input_layer].input, K.learning_phase()],
                                  [self.model.layers[output_layer].output])
                                  
    return _mrcnn_class
    
class MyCallback(keras.callbacks.Callback):

    def __init__(self): 

        return 
        
        # , pool_shape, image_shape, **kwargs):
        # super(PyramidROIAlign, self).__init__(**kwargs)
        # self.pool_shape = tuple(pool_shape)
        # self.image_shape = tuple(image_shape)

        
    def on_epoch_begin(self,epoch, logs = {}) :
        print('Start epoch {}  loss {}  {}\n'.format(epoch,logs,logs.keys()))
        return 

    def on_epoch_end  (self,epoch, logs = {}): 
        print('End   epoch {}  loss {}  {}\n'.format(epoch,logs,logs.keys()))
        return 

    def on_batch_begin(self,batch, logs = {}):
        print('\n... Start training of batch {} size {} keys {}'.
                format(batch,logs['size'],logs.keys()))
        return  
        
    def on_batch_end  (self,batch, logs = {}): 
        print('\n... End   training of batch {} logs {} keys {}'.
                                         format(batch,logs['loss'],logs.keys()))
        print('\n shape of output layer: {}'.format(tf.shape(self.model.layers[229].output)))
        return                                          
        
    def on_train_begin(self,logs = {}):        
        print('*** Start of Training {} '.format(time.time()))
        return 
        
    def on_train_end  (self,logs = {}):        
        print('*** End of Training   {} '.format(time.time()))    
        return 
