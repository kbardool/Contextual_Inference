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
# from collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow as tf
# import keras
import keras.backend as KB
# import keras.layers as KL
# import keras.initializers as KI
import keras.engine as KE
# import keras.models as KM
sys.path.append('..')
import mrcnn.utils as utils
import pprint

############################################################
##   
############################################################
def get_layer_output(model, model_input,output_layer, training_flag = True):
    _my_input = model_input 
    for name,inp in zip(model.input_names, model_input):
        print(' Input Name:  ({:24}) \t  Input shape: {}'.format(name, inp.shape))


    _mrcnn_class = KB.function(model.input , model.output)
#                               [model.keras_model.layers[output_layer].output])
    output = _mrcnn_class(_my_input)                  
    for name,out in zip (model.output_names,output):
        print(' Output Name: ({:24}) \t  Output shape: {}'.format(name, out.shape))
    return output
    
class PCTensor():
    """
    Subsamples proposals and generates target box refinment, class_ids, and masks for each.

    Inputs:
    -------    
    proposals:  [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
                be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes:   [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
                coordinates.
    gt_masks:   [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: 
    -------    
    Target ROIs and corresponding class IDs, bounding box shifts, and masks.
    tensor :            [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    stacked:            [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas:      [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,(dy, dx, log(dh), log(dw), class_id)]
                          Class-specific bbox refinments.
    target_mask:        [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                        Masks cropped to bbox boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, model, outputs= None):
        # super(DetectionTargetLayer, self).__init__(**kwargs)
        # super().__init__(**kwargs)
        self.config = model.config
        self.model  = model.keras_model
        self.mdl_outputs = outputs

    def build_gaussian_np(self):
        from scipy.stats import  multivariate_normal
        pp = pprint.PrettyPrinter(indent=2, width=100)

        img_h, img_w = self.config.IMAGE_SHAPE[:2]
        num_images   = self.config.BATCH_SIZE
        num_classes  = self.config.NUM_CLASSES  
        num_rois     = self.config.TRAIN_ROIS_PER_IMAGE
        #   print(bbox.shape)

        X = np.arange(0, img_w, 1)
        Y = np.arange(0, img_h, 1)
        X, Y = np.meshgrid(X, Y)
        pos = np.empty((num_rois,) + X.shape + (2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
        print(pos.shape)
        pos[:,:,:,0] = X;
        pos[:,:,:,1] = Y;

        # Build the covariance matrix
        pp1 = np.full((32), 12.0)
        pp2 = np.full((32), 19.0)
        cov  = np.stack((pp1,pp2),axis=-1)
        k_sess = KB.get_session()    
     
        
        # prt = self.pred_stacked
        Zout  = np.zeros((num_images, num_classes, img_w, img_h))
        print(' COVARIANCE SHAPE:',cov.shape) 
        # print('PRT SHAPES:', prt[0].shape, prt[1].shape)   

        for img in range(num_images):
            ps     = self.pred_stacked[img] # .eval(session = k_sess)  #     .eval(session=k_sess)
            print('shape of ps', ps.shape)
            print(ps)            
            for cls in range(num_classes):
                cls_idxs = np.argwhere(ps[:,6] == cls).squeeze() 
        #         ps = _ps[cls_idxs,:]        
                print('cls:',cls,' ',cls_idxs)
                width  = ps[:,5] - ps[:,3]
                height = ps[:,4] - ps[:,2]
                cx     = ps[:,3] + ( width  / 2.0)
                cy     = ps[:,2] + ( height / 2.0)
                means  = np.stack((cx,cy),axis = -1)
                print(type)
                print(ps.shape, type(ps),width.shape, height.shape, cx.shape, cy.shape, type(means),means.shape)
            
                rv  = list( map(multivariate_normal, means, cov))
                pdf = list( map(lambda x,y: x.pdf(y) , rv, pos))
                pdf_arr = np.asarray(pdf)
                print(pdf_arr.shape)
                pdf_sum = np.sum(pdf_arr[[cls_idxs]],axis=0)
                Zout[img,cls] += pdf_sum
        
        return Zout
        
    def get_pred_stacked(self):
        '''
        return all bboxes for images in a list, one ndarray per image 
        
        '''
        pred_stacked = []
        for img in range(self.config.BATCH_SIZE):
            _substack = np.empty((0,8),dtype=np.float32)
            for cls in range(self.config.NUM_CLASSES):
                # if  self.pred_cls_cnt[img, cls] > 0:
                    # _substack.append( self.pred_tensor[img, cls, 0:self.pred_cls_cnt[img, cls]] )   
                _substack = np.vstack((_substack, self.pred_tensor[img, cls, 0:self.pred_cls_cnt[img, cls]] ))   
            pred_stacked.append(np.asarray(_substack))                

            # self.pred_stacked.append(tf.concat(_substacked , 0))
        print('get stacked: pred_stacekd shape:',len(pred_stacked), pred_stacked[0].shape)    
        return pred_stacked
        
        
    def build_predictions(self, input = None):
        self.build_gt(input)
        # // pass model to TensorBuilder
        num_images  = self.config.BATCH_SIZE
        num_classes = self.config.NUM_CLASSES
        num_rois    = self.config.TRAIN_ROIS_PER_IMAGE
        num_max_gt  = self.config.DETECTION_MAX_INSTANCES
        num_cols    = 8 
        h, w        = self.config.IMAGE_SHAPE[:2]
        class_idx   = self.model.output_names.index('mrcnn_class')
        bbox_idx    = self.model.output_names.index('mrcnn_bbox')
        outroi_idx  = self.model.output_names.index('output_rois')
        
        if self.mdl_outputs == None:
            _mdl_outputs = get_layer_output(self.model,  input , 229, 1.0)
        
        self.mrcnn_class = _mdl_outputs[class_idx]
        self.mrcnn_bbox  = _mdl_outputs[bbox_idx]
        self.output_rois = _mdl_outputs[outroi_idx] * np.array([h,w,h,w])   
        # print('mrcnn_class idx: {}   mrcnn_bbox idx : {}   output_rois idx : {}'.format(class_idx, bbox_idx,outroi_idx))
        # print(' mrcnn_bbox : \n',self.mrcnn_bbox[0,0,:,:])
        # mdl_outputs[outroi_idx] returns the normalized coordinates, we multiply by h,w to get true coordinates
        
        _pred_arr        = np.zeros((num_images, num_classes, num_rois, num_cols ))      # img_in_batch, 4, 32, 8
        _pred_tensor     = np.zeros_like(_pred_arr)
        self.pred_stacked = []        

        self.pred_cls_cnt= np.zeros((num_images, num_classes), dtype='int16')
        # print('mrcnn_class shape : ', type(self.mrcnn_class), 'mrcnn_bbox.shape : ', type(self.mrcnn_bbox),\
              # 'output_rois.shape : ', self.output_rois.shape, 'pred_tensor shape : ', _pred_tensor.shape  )
        # print(self.output_rois)
        #---------------------------------------------------------------------------
        # use the argmaxof each row to determine the dominating (predicted) class
        #---------------------------------------------------------------------------
        _pred_class = np.argmax(self.mrcnn_class[:,:,:],axis=2).astype('int16')   # (32,)
        # print('mrcnn_class is: \n',self.mrcnn_class)
        # print('_pred_class is: \n',_pred_class.shape,'\n',_pred_class)
        
        for img in range(num_images):
            _substacked = []
            for cls in range(num_classes) :
                _class_idxs = np.argwhere( _pred_class[img,:] == cls )
                self.pred_cls_cnt[img,cls] = _class_idxs.shape[0] 
                # print('img/cls is: ' , img,'/',cls, '_class_idxs: ' , _class_idxs)
                for j , c_idx in enumerate(_class_idxs):
                    _pred_arr[img, cls, j,  0]  = j
                    _pred_arr[img, cls, j,  1]  = np.max(self.mrcnn_class[img, c_idx ])      # probability
                    _pred_arr[img, cls, j,2:6]  = self.output_rois[img,c_idx]                         # roi coordinates
                    _pred_arr[img, cls, j,  6]  = cls                                   # class_id
                    _pred_arr[img, cls, j,  7]  = c_idx                               # index from mrcnn_class array (temp for verification)               
                # sort each class in descending prediction order 
                order = _pred_arr[img,cls,:,1].argsort()
                _pred_arr[img, cls,:,1:] = _pred_arr[img, cls, order[::-1] ,1:]      #[img, cls,::-1]
                # _pred_tensor[img, cls,:,0]  = _pred_arr[img, cls,:,0]
                
            # for cls in range(0,num_classes):
                # if  self.pred_cls_cnt[img, cls] > 0:
                    # _substacked.append( _pred_arr[img, cls, 0:self.pred_cls_cnt[img, cls]] )   
            # self.pred_stacked.append(np.concatenate(_substacked,0))

        # self.pred_tensor = tf.convert_to_tensor(_pred_arr)
        self.pred_tensor = _pred_arr
        self.pred_stacked = self.get_pred_stacked()
        # print('pred_tensor type, shape :', type(self.pred_tensor), self.pred_tensor.shape)
        # for img in range(num_images):
            # print(self.pred_tensor[img].eval(session=KB.get_session()))
            # print(self.pred_tensor[img])
            # print('img ', img, ' substacked')
            # print(self.pred_stacked[img].eval(session=KB.get_session()))
        
        return 

    
    def build_gt(self, input):
        num_images   = self.config.BATCH_SIZE
        num_classes  = self.config.NUM_CLASSES        
        num_max_gt   = self.config.DETECTION_MAX_INSTANCES
        num_cols     = 8
        
        gtcls_idx = self.model.input_names.index('input_gt_class_ids')
        gtbox_idx = self.model.input_names.index('input_gt_boxes')
        gtmsk_idx = self.model.input_names.index('input_gt_masks')
        gt_classes = input[gtcls_idx]
        gt_bboxes  = input[gtbox_idx]

        _pred_arr  = np.zeros((num_images, num_classes, num_max_gt, num_cols ))      # img_in_batch, 4, 32, 8  
        self.gt_tensor  = np.zeros_like(_pred_arr)
        self.gt_cls_cnt = np.zeros((num_images, num_classes), dtype='int16')

        # gt_masks   = sample_x[gtmsk_idx][0,:,:,nz_idx]
        # gt_indexes = np.arange(gt_classes.shape[0],dtype='int16')
        # gt_probs   = np.ones(gt_classes.shape[0])
        # print('gt_classes.shape :',gt_classes.shape, 'gt_boxes.shape :',gt_bboxes.shape)

        
        for img in range(num_images):
            for cls in range(num_classes) :
                _class_idxs = np.argwhere( gt_classes[img, :] == cls)
                # print('k is: ' , k, '_class_idxs: ' , _class_idxs)
                self.gt_cls_cnt[img, cls] = _class_idxs.shape[0] 
                for j , c_idx in enumerate(_class_idxs):        
                    self.gt_tensor[img, cls, j,  0]  = j
                    self.gt_tensor[img, cls, j,  1]  = 1.0                                 # probability
                    self.gt_tensor[img, cls, j, 2:6] = gt_bboxes[img,c_idx,:]                         # roi coordinates
                    self.gt_tensor[img, cls, j,  6]  = cls                                 # class_id
                    self.gt_tensor[img, cls, j,  7]  = c_idx                               # index from mrcnn_class array (temp for verification)

                  
        self.gt_stacked = []
                
        for img in range(num_images):
            _substacked = np.empty((0,8))
            for cls in range(0,num_classes):
                if self.gt_cls_cnt[img, cls] > 0:
                    _substacked = np.vstack((_substacked, self.gt_tensor[img , cls, 0:self.gt_cls_cnt[img, cls]] ))
            self.gt_stacked.append( _substacked )   

        # print('gt_tensor : (idx, class, prob, y1, x1, y2, x2)', self.gt_tensor.shape, '\n')
        # for i in range(len(self.gt_stacked)):
            # print(self.gt_stacked[i].shape)
            # print(self.gt_stacked[i])
            
    def get_gt_stacked(self):
        '''
        return all bboxes for images in a list, one ndarray per image 
        
        '''
        self.gt_stacked = []
                
        for img in range(self.config.BATCH_SIZE):
            _substacked = np.empty((0,8))
            for cls in range(self.config.NUM_CLASSES):
                if self.gt_cls_cnt[img, cls] > 0:
                    _substacked = np.vstack((_substacked, self.gt_tensor[img , cls, 0:self.gt_cls_cnt[img, cls]] ))
            self.gt_stacked.append( _substacked )   



    def __repr__(self):
        print(' I\'m in repr ...!')
        
      
    def __str__(self):
        print(' I\'m in __str__') 

        
        
from scipy.stats import  multivariate_normal
import numpy as np
def bbox_gaussian( bbox, Zin ):
    """
    receive a bounding box, and generate a gaussian distribution centered on the bounding box and with a 
    covariance matrix based on the width and height of the bounding box/. 
    Inputs : 
    --------
    bbox :  (index, class_prob, y1, x1, y2, x2, class_id, old_idx)
    bbox :  (index, class_id, class_prob, cx, cy, width, height)
    Returns:
    --------
    bbox_g  grid mesh [image_height, image width] covering the distribution

    """
    img_w, img_h  = Zin.shape

    width  = bbox[5] - bbox[3]
    height = bbox[4] - bbox[2]
    cx     = bbox[3] + ( width  / 2.0)
    cy     = bbox[2] + ( height / 2.0)
    
#     cx, cy, width, height = bbox[3:]
    print('center is ({:4f},{:4f}) width: {:4f}  height: {:4f} '.format(cx, cy, width,  height))
#     srtd_cpb_2 = np.column_stack((srtd_cpb[:, 0:2], cx,cy, width, height ))
    X = np.arange(0, img_w, 1)
    Y = np.arange(0, img_h, 1)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,0] = X;
    pos[:,:,1] = Y;

    rv = multivariate_normal([cx,cy],[[12,0.0] , [0.0,19]])
    return  rv.pdf(pos)

    
# def build_gaussian_independent( pc_tensor ):
    # """
    # using the prediction tensor, generate a gaussian distribution centered on the bounding box and with a 
    # covariance matrix based on the width and height of the bounding box/. 
    # Inputs : 
    # --------
    # tensor [num_images, num_classes, (idx, class_prob, y1, x1, y2, x2, class_id, old_idx)]

    # Returns:
    # --------
    # Zout   [num_images, num_classes, image_width, image_height]

    # """
    # img_h, img_w = pc_tensor.config.IMAGE_SHAPE[:2]
    # num_images   = pc_tensor.config.BATCH_SIZE
    # num_classes  = pc_tensor.config.NUM_CLASSES  
    
# #   print(bbox.shape)
    # width  = pc_tensor.pred_tensor[:,:,:,5] - pc_tensor.pred_tensor[:,:,:,3]
    # height = pc_tensor.pred_tensor[:,:,:,4] - pc_tensor.pred_tensor[:,:,:,2]
    # cx     = pc_tensor.pred_tensor[:,:,:,3] + ( width  / 2.0)
    # cy     = pc_tensor.pred_tensor[:,:,:,2] + ( height / 2.0)
    # means = np.stack((cx,cy),axis = -1)
    # Zout  = np.zeros((num_images, num_classes, img_w, img_h))

# #   srtd_cpb_2 = np.column_stack((srtd_cpb[:, 0:2], cx,cy, width, height ))
    # X = np.arange(0, img_w, 1)
    # Y = np.arange(0, img_h, 1)
    # X, Y = np.meshgrid(X, Y)
    # pos = np.empty((num_images, num_classes,) + X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    # pos[:,:,:,:,0] = X;
    # pos[:,:,:,:,1] = Y;

    # for img in range(num_images):
        # for cls in range(num_classes):
            # _cnt = pc_tensor.pred_cls_cnt[img,cls]
            # print('class id: ', cls, 'class count: ',_cnt)
            # for box in range(_cnt):
                
                # mns = means[img,cls,box]
                # print('** bbox is : ' ,pc_tensor.pred_tensor[img,cls,box])
                # print('    center is ({:4f},{:4f})  width is {:4f} height is {:4f} '\
                    # .format(mns[0],mns[1],width[img,cls,box],height[img,cls,box]))            
                
                # rv = multivariate_normal(mns,[[12,0.0] , [0.0,19]])
                # Zout[img,cls,:,:] += rv.pdf(pos[img,cls])
                
    # return Zout

    # def build_gaussian_OLD(self):
        # """
        # using the prediction tensor, generate a gaussian distribution centered on the bounding box and with a 
        # covariance matrix based on the width and height of the bounding box/. 
        # Inputs : 
        # --------
        # tensor [num_images, num_classes, (idx, class_prob, y1, x1, y2, x2, class_id, old_idx)]

        # Returns:
        # --------
        # Zout   [num_images, num_classes, image_width, image_height]

        # """
        # img_h, img_w = self.config.IMAGE_SHAPE[:2]
        # num_images   = self.config.BATCH_SIZE
        # num_classes  = self.config.NUM_CLASSES  
        
    # #   print(bbox.shape)
        # width  = self.pred_tensor[:,:,:,5] - self.pred_tensor[:,:,:,3]
        # height = self.pred_tensor[:,:,:,4] - self.pred_tensor[:,:,:,2]
        # cx     = self.pred_tensor[:,:,:,3] + ( width  / 2.0)
        # cy     = self.pred_tensor[:,:,:,2] + ( height / 2.0)
        # means = np.stack((cx,cy),axis = -1)
        # Zout  = np.zeros((num_images, num_classes, img_w, img_h))

    # #   srtd_cpb_2 = np.column_stack((srtd_cpb[:, 0:2], cx,cy, width, height ))
        # X = np.arange(0, img_w, 1)
        # Y = np.arange(0, img_h, 1)
        # X, Y = np.meshgrid(X, Y)
        # pos = np.empty((num_images, num_classes,) + X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
        # pos[:,:,:,:,0] = X;
        # pos[:,:,:,:,1] = Y;

        # for img in range(num_images):
            # for cls in range(num_classes):
                # _cnt = self.pred_cls_cnt[img,cls]

                # for box in range(_cnt):
                    
                    # mns = means[img,cls, 0 : _cnt]
                    # print('img: ',img, 'class: ', cls, 'class count: ',_cnt, 'shape of mns :',mns.shape)
                    # # print('** bbox is : ' ,self.pred_tensor[img,cls,box])
                    # # print('    center is ({:4f},{:4f})  width is {:4f} height is {:4f} '\
                        # # .format(mns[0],mns[1],width[img,cls,box],height[img,cls,box]))            
                    # # fn = lambda x: multivariate_normal(x, [[12,0.0] , [0.0,19]])
                    # # rv = tf.map_fn(fn, 
                    # rv = np.apply_along_axis(multivariate_normal, 1, mns, [[12,0.0] , [0.0,19]])
                    # print('rv :',rv.shape, rv)
                    # _zo = rv[:].pdf(pos[img,cls])
                    # print('zo :',_zo.shape)
                    # # Zout[img,cls,:,:] += rv.pdf(pos[img,cls])
                    
        # return Zout
