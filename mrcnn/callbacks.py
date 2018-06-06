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
import tensorflow    as tf
import keras
import keras.backend as KB
import keras.layers  as KL
import keras.initializers as KI
import keras.engine  as KE
import keras.models  as KM
import pprint 
pp = pprint.PrettyPrinter(indent=4, width=100)

# def get_layer_output(model, output_layer, model_input, training_flag = True):
    # _mrcnn_class = KB.function([model.input]+[KB.learning_phase()],
                              # [model.layers[output_layer].output])
    # output = _mrcnn_class([model_input,training_flag])[0]                                
    # return output
    
    
def get_layer_output_1(model, model_input, requested_layers, training_flag = True, verbose = True):
    _my_input = model_input + [training_flag]
    if verbose: 
        print('/* Inputs */')
        for i, (input_layer,input) in enumerate(zip(model.input, model_input)):
            print('Input {:2d}:  ({:40s}) \t  Input shape: {}'.format(i, input_layer.name, input.shape))
            
    requested_layers_list = [model.outputs[x] for x in requested_layers]
    
    _mrcnn_class = KB.function(model.input, requested_layers_list)   
    results = _mrcnn_class(model_input)                    
    
    if verbose:
        print('\n/* Outputs */')    
        for line, (i, output_layer, output) in  enumerate(zip (requested_layers, requested_layers_list , results)):
            print('Output idx: {:2d}    Layer: {:2d}: ({:40s}) \t  Output shape: {}'.format(line, i, output_layer.name, output.shape))
        print('\nNumber of layers generated: ', len(results), '\n')

        for line, (i, output_layer, output) in  enumerate(zip (requested_layers, requested_layers_list , results)):
            m = re.fullmatch(r"^.*/(.*):.*$", output_layer.name )
            varname  = m.group(1) if m else "<varname>"
            print('{:25s} = model_output[{:d}]          # layer: {:2d}   shape: {}' .format(varname, line, i, output.shape ))
    return results

    
    
def get_layer_output_2(model, model_input, training_flag = True, verbose = True):
    if verbose: 
        print('/* Inputs */')
        for i, (input_layer,input) in enumerate(zip(model.input, model_input)):
            print('Input {:2d}:  ({:40s}) \t  Input shape: {}'.format(i, input_layer.name, input.shape))

    get_output = KB.function(model.input , model.outputs)
    results = get_output(model_input)                  
    
    print(type(results))
    # if verbose:
        # print('\n/* Outputs */')    
        # for i, (output_layer, output) in enumerate (zip (model.outputs , model_output)):
            # print('Output {:2d}: ({:40s}) \t  Output shape: {}'.format(i, output_layer.name, output.shape))
            
            
    if verbose:
        print('\n/* Outputs */')    
        for line, ( output_layer, output) in  enumerate(zip (model.outputs, results)):
            print('Output idx: {:2d}    Layer: {:2d}: ({:40s}) \t  Output shape: {}'.format(line, line, output_layer.name, output.shape))
        print('\nNumber of layers generated: ', len(results), '\n')

        for line, ( output_layer, output) in  enumerate(zip (model.outputs, results)):
            m = re.fullmatch(r"^.*/(.*):.*$", output_layer.name )
            varname  = m.group(1) if m else "<varname>"
            print('{:25s} = model_output[{:d}]          # layer: {:2d}   shape: {}' .format(varname, line, line, output.shape ))            
    return results

    
class MyCallback(keras.callbacks.Callback):

    def __init__(self): 

        return 
        
        # , pool_shape, image_shape, **kwargs):
        # super(PyramidROIAlign, self).__init__(**kwargs)
        # self.pool_shape = tuple(pool_shape)
        # self.image_shape = tuple(image_shape)

    def on_epoch_begin(self, epoch, logs = {}) :
        print('\n>>> Start epoch {}  \n'.format(epoch))
        return 

    def on_epoch_end  (self, epoch, logs = {}): 
        print('\n>>>End   epoch {}  \n'.format(epoch))
        return 

    def on_batch_begin(self, batch, logs = {}):
        print('\n... Start training of batch {} size {} '.format(batch,logs['size']))
        pp.pprint(self.model._feed_inputs)
        k_sess = KB.get_session()
        # self.model._feed_inputs[1].eval(session=k_sess)
        return  
        
    def on_batch_end  (self, batch, logs = {}): 
        print('\n... End   training of batch {} '.format(batch,logs['loss']))
        pp.pprint(logs)
        return                                          
        
    def on_train_begin(self,logs = {}):        
        pp = pprint.PrettyPrinter(indent=4)
        print('\n *****  Start of Training {} '.format(time.time()))
        return 
        
    def on_train_end  (self,logs = {}):        
        pp = pprint.PrettyPrinter(indent=4)  
        print('\n\n')
        print('***** End of Training   {} '.format(time.time()))    
        return 
