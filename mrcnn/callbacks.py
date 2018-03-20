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
import pprint 

# def get_layer_output(model, output_layer, model_input, training_flag = True):
    # _mrcnn_class = K.function([model.input]+[K.learning_phase()],
                              # [model.layers[output_layer].output])
    # output = _mrcnn_class([model_input,training_flag])[0]                                
    # return output
    
    
def get_layer_output(model, model_input, output_layer, training_flag = True):
    # print(type(training_flag))

    _my_input = model_input + [training_flag]
    print('input type is',type(_my_input))
    for ind, i in enumerate(_my_input):
        print('model_input {}  type {}'.format(ind, type(i)))

    _mrcnn_class = K.function(model.input, [model.layers[output_layer].output])
    output = _mrcnn_class(_my_input)[0]                                
    return output


    
class MyCallback(keras.callbacks.Callback):

    def __init__(self): 

        return 
        
        # , pool_shape, image_shape, **kwargs):
        # super(PyramidROIAlign, self).__init__(**kwargs)
        # self.pool_shape = tuple(pool_shape)
        # self.image_shape = tuple(image_shape)

    def on_epoch_begin(self,epoch, logs = {}) :
        print('Start epoch {}  \n'.format(epoch))
        pp = pprint.PrettyPrinter(indent=4)
        return 

    def on_epoch_end  (self,epoch, logs = {}): 
        print('End   epoch {}  \n'.format(epoch))
        pp = pprint.PrettyPrinter(indent=4)        
        return 

    def on_batch_begin(self,batch, logs = {}):
        # print('\n... Start training of batch {} size {} '.format(batch,logs['size']))
        
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.model._feed_inputs)
        return  
        
    def on_batch_end  (self,batch, logs = {}): 
        # print('\n... End   training of batch {} '.format(batch,logs['loss']))
        # pp = pprint.PrettyPrinter(indent=4)
        # i = 229
        # print('\n shape of output layer: {} '.format(i)) ## , tf.shape(self.model.layers[i].output)))
        # for i in (self.model.input):
            # print('input  type: {}'.format(i.get_shape()))
        # print(self.model.layers[229].output.eval())
        # layer_out = get_layer_output(self.model, 229, self.model.input, 1)
        # print('type of layer out is {} shape is {}'.format(type(layer_out), layer_out.shape))
        return                                          
        
    def on_train_begin(self,logs = {}):        
        pp = pprint.PrettyPrinter(indent=4)
        i = 229
        pp.pprint(self.model.layers[i].__dict__)  
        pp.pprint(self.model.layers[i]._inbound_nodes[0].__dict__)  
        pp.pprint(self.model.layers[i].layer.__dict__)  
        print('size of input {} type of input {}'.format(len(self.model.input), type(self.model.input)))

        print('*** Start of Training {} '.format(time.time()))
        return 
        
    def on_train_end  (self,logs = {}):        
        pp = pprint.PrettyPrinter(indent=4)  
        # pp.pprint(self.model.__dict__)
        print('\n'*3)
        # pp.pprint(dir(self.model))
        print('*** End of Training   {} '.format(time.time()))    
        return 
