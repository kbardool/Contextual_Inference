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
# import scipy.misc
import tensorflow as tf
import numpy as np
# import keras
import keras.backend as KB
# import keras.layers as KL
# import keras.initializers as KI
import keras.engine as KE
# import keras.models as KM
sys.path.append('..')
import mrcnn.utils as utils


############################################################
#   
############################################################
def get_layer_output(model, model_input,output_layer, training_flag = True):
    _my_input = model_input 
    for name,inp in zip(model.input_names, model_input):
        print(' Input Name:  ({:24}) \t  Input shape: {}'.format(name, inp.shape))


    _mrcnn_class = KB.function(model.input , model.output)
#                               [model.keras_model.layers[output_layer].output])
    output = _mrcnn_class(_my_input)                  
    
    for name,out in zip (model.output_names,output):
        print(' Output Name: ({:24}) \t Output shape: {}'.format(name, out.shape))
    return output
    
class PCTensor():
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

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
        print('mrcnn_class idx: {}   mrcnn_bbox idx : {}   output_rois idx : {}'.format(class_idx, bbox_idx,outroi_idx))
        
        if self.mdl_outputs == None:
            _mdl_outputs = get_layer_output(self.model,  input , 229, 1.0)
        
        self.mrcnn_class = _mdl_outputs[class_idx]
        self.mrcnn_bbox  = _mdl_outputs[bbox_idx]
        self.output_rois = _mdl_outputs[outroi_idx] * np.array([h,w,h,w])   
        
        # mdl_outputs[outroi_idx] returns the normalized coordinates, we multiply by h,w to get true coordinates
        _pred_arr        = np.zeros((num_images, num_classes, num_rois, num_cols ))      # img_in_batch, 4, 32, 8
        self.pred_tensor      = np.zeros_like(_pred_arr)
        self.pred_cls_cnt= np.zeros((num_images, num_classes), dtype='int16')
        print('mrcnn_class shape   : ', self.mrcnn_class.shape, 'mrcnn_bbox.shape : ', self.mrcnn_bbox.shape,\
                'output_rois.shape : ', self.output_rois.shape)
        print('Tensor shape is     : ', self.pred_tensor.shape  )
        
        #---------------------------------------------------------------------------
        # use the argmaxof each row to determine the dominating (predicted) class
        #---------------------------------------------------------------------------
        _pred_class = np.argmax(self.mrcnn_class[:,:,:],axis=2).astype('int16')   # (32,)
        # print('mrcnn_class is: \n',mrcnn_class)
        # print('_pred_class is: \n',_pred_class)
        
        for img in range(num_images):
            for cls in range(num_classes) :
                _class_idxs = np.argwhere( _pred_class[img,:] == cls )
                # print('img is: ' , img, '_class_idxs: ' , _class_idxs)
                self.pred_cls_cnt[img,cls] = _class_idxs.shape[0] 
                for j , c_idx in enumerate(_class_idxs):
                    _pred_arr[img, cls, j,  0]  = j
                    _pred_arr[img, cls, j,  1]  = np.max(self.mrcnn_class[img, c_idx ])      # probability
                    _pred_arr[img, cls, j,2:6]  = self.output_rois[img,c_idx]                         # roi coordinates
                    _pred_arr[img, cls, j,  6]  = cls                                   # class_id
                    _pred_arr[img, cls, j,  7]  = c_idx                               # index from mrcnn_class array (temp for verification)
                    
                    
        # sort each class in descending prediction order 

        order = _pred_arr[:,:,:,1].argsort()

        for img in range(num_images):
            for cls in range(num_classes):
                self.pred_tensor[img, cls,:,1:] =  _pred_arr[img, cls ,order[img, cls,::-1],1:]      
            self.pred_tensor[img, :,:,0] = _pred_arr[img, :,:,0]

        print('tensor shape', self.pred_tensor.shape)
        
        self.stack_predictions()

        return 
    
    
    def build_gt(self, input):
        num_images   = self.config.BATCH_SIZE
        num_classes  = self.config.NUM_CLASSES        
        num_max_gt   = self.config.DETECTION_MAX_INSTANCES
        num_cols     = 8
        
        gtcls_idx = self.model.input_names.index('input_gt_class_ids')
        gtbox_idx = self.model.input_names.index('input_gt_boxes')
        gtmsk_idx = self.model.input_names.index('input_gt_masks')
        print('gtcls_idx: ',gtcls_idx, 'gtbox_idx :', gtbox_idx)
        gt_classes = input[gtcls_idx]
        gt_bboxes  = input[gtbox_idx]

        _pred_arr  = np.zeros((num_images, num_classes, num_max_gt, num_cols ))      # img_in_batch, 4, 32, 8  
        self.gt_tensor  = np.zeros_like(_pred_arr)
        self.gt_cls_cnt = np.zeros((num_images, num_classes), dtype='int16')

        # gt_masks   = sample_x[gtmsk_idx][0,:,:,nz_idx]
        # gt_indexes = np.arange(gt_classes.shape[0],dtype='int16')
        # gt_probs   = np.ones(gt_classes.shape[0])

        print('gt_classes.shape :',gt_classes.shape, 'gt_boxes.shape :',gt_bboxes.shape)

        
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
            for cls in range(1,num_classes):
                if self.gt_cls_cnt[img, cls] > 0:
                    _substacked = np.vstack((_substacked, self.gt_tensor[img , cls, 0:self.gt_cls_cnt[img, cls]] ))
            self.gt_stacked.append( _substacked )   

        print('gt_tensor : (idx, class, prob, y1, x1, y2, x2)', self.gt_tensor.shape, '\n')
        for i in range(len(self.gt_stacked)):
            print(self.gt_stacked[i].shape)
            print(self.gt_stacked[i])
        
    def stack_predictions(self):
        # build stacked tensor 
        num_images   = self.config.BATCH_SIZE
        num_classes  = self.config.NUM_CLASSES        
        # print('pc_tensor.pred_cls_cnt.shape', self.pred_cls_cnt.shape)
        # print(self.pred_cls_cnt)        

        self.pred_stacked = []        
        for img in range(num_images):
            _substacked = np.empty((0,8))
            for cls in range(1,num_classes):
                if self.pred_cls_cnt[img, cls] > 0:
                    _substacked = np.vstack((_substacked, self.pred_tensor[img, cls, 0:self.pred_cls_cnt[img, cls]] ))   
            self.pred_stacked.append( _substacked )   
        return self.pred_stacked

        
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

    
def build_gaussian( pc_tensor ):
    """
    using the prediction tensor, generate a gaussian distribution centered on the bounding box and with a 
    covariance matrix based on the width and height of the bounding box/. 
    Inputs : 
    --------
    tensor [num_images, num_classes, (idx, class_prob, y1, x1, y2, x2, class_id, old_idx)]

    Returns:
    --------
    Zout   [num_images, num_classes, image_width, image_height]

    """
    img_h, img_w = pc_tensor.config.IMAGE_SHAPE[:2]
    num_images   = pc_tensor.config.BATCH_SIZE
    num_classes  = pc_tensor.config.NUM_CLASSES  
    
#   print(bbox.shape)
    width  = pc_tensor.pred_tensor[:,:,:,5] - pc_tensor.pred_tensor[:,:,:,3]
    height = pc_tensor.pred_tensor[:,:,:,4] - pc_tensor.pred_tensor[:,:,:,2]
    cx     = pc_tensor.pred_tensor[:,:,:,3] + ( width  / 2.0)
    cy     = pc_tensor.pred_tensor[:,:,:,2] + ( height / 2.0)
    means = np.stack((cx,cy),axis = -1)
    Zout  = np.zeros((num_images, num_classes, img_w, img_h))

#   srtd_cpb_2 = np.column_stack((srtd_cpb[:, 0:2], cx,cy, width, height ))
    X = np.arange(0, img_w, 1)
    Y = np.arange(0, img_h, 1)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty((num_images, num_classes,) + X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,:,:,0] = X;
    pos[:,:,:,:,1] = Y;

    for img in range(num_images):
        for cls in range(num_classes):
            _cnt = pc_tensor.pred_cls_cnt[img,cls]
            print('class id: ', cls, 'class count: ',_cnt)
            for box in range(_cnt):
                
                mns = means[img,cls,box]
                print('** bbox is : ' ,pc_tensor.pred_tensor[img,cls,box])
                print('    center is ({:4f},{:4f})  width is {:4f} height is {:4f} '\
                    .format(mns[0],mns[1],width[img,cls,box],height[img,cls,box]))            
                
                rv = multivariate_normal(mns,[[12,0.0] , [0.0,19]])
                Zout[img,cls,:,:] += rv.pdf(pos[img,cls])
                
    return Zout

    
    
# def overlaps_graph(boxes1, boxes2):
    # """
    # Computes IoU overlaps between two sets of boxes.
    # boxes1, boxes2: [N, (y1, x1, y2, x2)].
    # """
    # # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # # every boxes1 against every boxes2 without loops.
    # # TF doesn't have an equivalent to np.repeat() so simulate it
    # # using tf.tile() and tf.reshape.
    # b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            # [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    # print(tf.shape(b1))
    # b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # print(tf.shape(b2))

    # # 2. Compute intersections
    # b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    # b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    # y1 = tf.maximum(b1_y1, b2_y1)
    # x1 = tf.maximum(b1_x1, b2_x1)
    # y2 = tf.minimum(b1_y2, b2_y2)
    # x2 = tf.minimum(b1_x2, b2_x2)
    # intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    # # 3. Compute unions
    # b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    # b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    # union = b1_area + b2_area - intersection
    
    # # 4. Compute IoU and reshape to [boxes1, boxes2]
    # iou = intersection / union
    # overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    # print('Shape of overlaps',tf.shape(overlaps))
    # return overlaps


# def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    # """
    # Generates detection targets for one image. Subsamples proposals and
    # generates target class IDs, bounding box deltas, and masks for each.

    # Inputs:
    # -------
    # proposals:          [N, (y1, x1, y2, x2)] in normalized coordinates. 
                        # Might be zero padded if there are not enough proposals.
    # gt_class_ids:       [MAX_GT_INSTANCES] int class IDs
    # gt_boxes:           [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    # gt_masks:           [height, width, MAX_GT_INSTANCES] of boolean type.

    # Returns:            Target ROIs and corresponding class IDs, bounding box shifts, and masks.
    # --------
    # rois:               [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    # class_ids:          [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    # deltas:             [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                        # Class-specific bbox refinments.
    # masks:              [TRAIN_ROIS_PER_IMAGE, height, width). Masks cropped to bbox
                        # boundaries and resized to neural network output size.

    # Note: Returned arrays might be zero padded if not enough target ROIs.
    # """
    # # Assertions
    # asserts = [
        # tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion"),
    # ]
    
    # with tf.control_dependencies(asserts):
        # proposals = tf.identity(proposals)

    # # Remove zero padding
    
    # proposals, _        = utils.trim_zeros_graph(proposals, name="trim_proposals")
    # gt_boxes, non_zeros = utils.trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    # gt_class_ids        = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
    # gt_masks            = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,name="trim_gt_masks")

    # #------------------------------------------------------------------------------------------
    # # Handle COCO crowds
    # # A crowd box in COCO is a bounding box around several instances. Exclude
    # # them from training. A crowd box is given a negative class ID.
    # #------------------------------------------------------------------------------------------
    # # tf.where : returns the coordinates of true elements of  the specified conditon.
    # #            The coordinates are returned in a 2-D tensor where the first dimension (rows) 
    # #            represents the number of true elements, and the second dimension (columns) 
    # #            represents the coordinates of the true elements. 
    # #            Keep in mind, the shape of the output tensor can vary depending on how many 
    # #            true values there are in input. Indices are output in row-major order.

    # # tf.gather: Gather slices from params axis (default = 0) according to indices.
    # #            indices must be an integer tensor of any dimension (usually 0-D or 1-D). 
    # #            Produces an output tensor with shape:
    # #                   params.shape[:axis] + indices.shape + params.shape[axis + 1:] 
    
    # # tf.squeeze: Removes dimensions of size 1 from the shape of a tensor.
    # #            Given a tensor input, this operation returns a tensor of the same type with 
    # #            all dimensions of size 1 removed. If you don't want to remove all size 1 
    # #            dimensions, you can remove specific size 1 dimensions by specifying axis.
    
    # crowd_ix     = tf.where(gt_class_ids < 0)[:, 0]
    # non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    # crowd_boxes  = tf.gather(gt_boxes, crowd_ix)
    # crowd_masks  = tf.gather(gt_masks, crowd_ix, axis=2)
    # gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    # gt_boxes     = tf.gather(gt_boxes, non_crowd_ix)
    # gt_masks     = tf.gather(gt_masks, non_crowd_ix, axis=2)


    # # Compute overlaps with crowd boxes [anchors, crowds]
    # crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    # crowd_iou_max  = tf.reduce_max(crowd_overlaps, axis=1)
    # no_crowd_bool  = (crowd_iou_max < 0.001)


    # # Compute overlaps matrix [proposals, gt_boxes] - The IoU between 
    # # proposals anf gt_boxes (non-crowd gt boxes, designated by classId < 0 in Coco)
    # overlaps = overlaps_graph(proposals, gt_boxes)
    # roi_iou_max = tf.reduce_max(overlaps, axis=1)

    # # Determine postive and negative ROIs
    # # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    # positive_roi_bool = (roi_iou_max >= 0.5)
    # positive_indices  = tf.where(positive_roi_bool)[:, 0]

    # # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    # negative_indices  = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # # Subsample ROIs. Aim for 33% positive (config.ROI_POSITIVE_RATIO = 0.33)
    # # Positive ROIs   33% of config.TRAIN_ROIS_PER_IMAGE ~  11
    # positive_count   = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    # positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    # positive_count   = tf.shape(positive_indices)[0]
    
    # # Negative ROIs. Add enough to maintain positive:negative ratio.
    # # negative_count = int((positive_count / config.ROI_POSITIVE_RATIO) - positive_count)
        
    # r = 1.0 / config.ROI_POSITIVE_RATIO
    # negative_count   = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    # negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    
    # # Gather selected ROIs
    # positive_rois = tf.gather(proposals, positive_indices)
    # negative_rois = tf.gather(proposals, negative_indices)

    # # Assign positive ROIs to GT boxes.
    # positive_overlaps     = tf.gather(overlaps, positive_indices)
    # roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
    # roi_gt_boxes          = tf.gather(gt_boxes    , roi_gt_box_assignment)
    # roi_gt_class_ids      = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # # Compute bbox refinement for positive ROIs
    # deltas  = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    # deltas /= config.BBOX_STD_DEV

    # # Assign positive ROIs to GT masks
    # # Permute masks to [N, height, width, 1]
    
    # transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    
    # # Pick the right mask for each ROI
    # roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # # Compute mask targets
    # boxes = positive_rois
    
    # if config.USE_MINI_MASK:
        # # Transform ROI corrdinates from normalized image space
        # # to normalized mini-mask space.
        # y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        # gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        # gt_h = gt_y2 - gt_y1
        # gt_w = gt_x2 - gt_x1
        # y1 = (y1 - gt_y1) / gt_h
        # x1 = (x1 - gt_x1) / gt_w
        # y2 = (y2 - gt_y1) / gt_h
        # x2 = (x2 - gt_x1) / gt_w
        # boxes = tf.concat([y1, x1, y2, x2], 1)
    
    # box_ids = tf.range(0, tf.shape(roi_masks)[0])
    # masks   = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     # box_ids,
                                     # config.MASK_SHAPE)
                                     
    # # Remove the extra dimension from masks.
    # masks = tf.squeeze(masks, axis=3)

    # # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # # binary cross entropy loss.
    # masks = tf.round(masks)

    # # Append negative ROIs and pad bbox deltas and masks that
    # # are not used for negative ROIs with zeros.
    # rois             = tf.concat([positive_rois, negative_rois], axis=0)
    # N                = tf.shape(negative_rois)[0]
    # P                = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    # rois             = tf.pad(rois, [(0, P), (0, 0)])
    # roi_gt_boxes     = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    # roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    # deltas           = tf.pad(deltas, [(0, N + P), (0, 0)])
    # masks            = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    # return rois, roi_gt_class_ids, deltas, masks












        
        # # Slice the batch and run a graph for each slice    
        # # TODO: Rename target_bbox to target_deltas for clarity
        
        # names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        # outputs = utils.batch_slice([proposals, gt_class_ids, gt_boxes, gt_masks],
                                    # lambda w, x, y, z: detection_targets_graph(w, x, y, z, self.config),
                                    # self.config.IMAGES_PER_GPU, names=names)
        # return outputs

    # def compute_output_shape(self, input_shape):
        # return [
            # (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            # (None, 1),  # class_ids
            # (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            # (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             # self.config.MASK_SHAPE[1])  # masks
        # ]

    # def compute_mask(self, inputs, mask=None):
        # return [None, None, None, None]


# ############################################################
# #  Detection Layer
# ############################################################

# def clip_to_window(window, boxes):
    # """
    # window: (y1, x1, y2, x2). The window in the image we want to clip to.
    # boxes: [N, (y1, x1, y2, x2)]
    # """
    # boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
    # boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
    # boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
    # boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
    # return boxes


# def refine_detections(rois, probs, deltas, window, config):
    # """Refine classified proposals and filter overlaps and return final
    # detections.

    # Inputs:
        # rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        # probs: [N, num_classes]. Class probabilities.
        # deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                # bounding box deltas.
        # window: (y1, x1, y2, x2) in image coordinates. The part of the image
            # that contains the image excluding the padding.

    # Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    # """
    # # Class IDs per ROI
    # class_ids = np.argmax(probs, axis=1)
    # # Class probability of the top class of each ROI
    # class_scores = probs[np.arange(class_ids.shape[0]), class_ids]
    # # Class-specific bounding box deltas
    # deltas_specific = deltas[np.arange(deltas.shape[0]), class_ids]
    # # Apply bounding box deltas
    # # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    # refined_rois = utils.apply_box_deltas(
        # rois, deltas_specific * config.BBOX_STD_DEV)
    # # Convert coordiates to image domain
    # # TODO: better to keep them normalized until later
    # height, width = config.IMAGE_SHAPE[:2]
    # refined_rois *= np.array([height, width, height, width])
    # # Clip boxes to image window
    # refined_rois = clip_to_window(window, refined_rois)
    # # Round and cast to int since we're deadling with pixels now
    # refined_rois = np.rint(refined_rois).astype(np.int32)

    # # TODO: Filter out boxes with zero area

    # # Filter out background boxes
    # keep = np.where(class_ids > 0)[0]
    # # Filter out low confidence boxes
    # if config.DETECTION_MIN_CONFIDENCE:
        # keep = np.intersect1d(
            # keep, np.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[0])

    # # Apply per-class NMS
    # pre_nms_class_ids = class_ids[keep]
    # pre_nms_scores = class_scores[keep]
    # pre_nms_rois = refined_rois[keep]
    # nms_keep = []
    # for class_id in np.unique(pre_nms_class_ids):
        # # Pick detections of this class
        # ixs = np.where(pre_nms_class_ids == class_id)[0]
        # # Apply NMS
        # class_keep = utils.non_max_suppression(
            # pre_nms_rois[ixs], pre_nms_scores[ixs],
            # config.DETECTION_NMS_THRESHOLD)
        # # Map indicies
        # class_keep = keep[ixs[class_keep]]
        # nms_keep = np.union1d(nms_keep, class_keep)
    # keep = np.intersect1d(keep, nms_keep).astype(np.int32)

    # # Keep top detections
    # roi_count = config.DETECTION_MAX_INSTANCES
    # top_ids = np.argsort(class_scores[keep])[::-1][:roi_count]
    # keep = keep[top_ids]

    # # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # # Coordinates are in image domain.
    # result = np.hstack((refined_rois[keep],
                        # class_ids[keep][..., np.newaxis],
                        # class_scores[keep][..., np.newaxis]))
    # return result


# class DetectionLayer(KE.Layer):
    # """Takes classified proposal boxes and their bounding box deltas and
    # returns the final detection boxes.

    # Returns:
    # [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    # """

    # def __init__(self, config=None, **kwargs):
        # super(DetectionLayer, self).__init__(**kwargs)
        # self.config = config

    # def call(self, inputs):
        # def wrapper(rois, mrcnn_class, mrcnn_bbox, image_meta):
            # detections_batch = []
            # for b in range(self.config.BATCH_SIZE):
                # _, _, window, _ =  parse_image_meta(image_meta)
                # detections = refine_detections(
                    # rois[b], mrcnn_class[b], mrcnn_bbox[b], window[b], self.config)
                # # Pad with zeros if detections < DETECTION_MAX_INSTANCES
                # gap = self.config.DETECTION_MAX_INSTANCES - detections.shape[0]
                # assert gap >= 0
                # if gap > 0:
                    # detections = np.pad(
                        # detections, [(0, gap), (0, 0)], 'constant', constant_values=0)
                # detections_batch.append(detections)

            # # Stack detections and cast to float32
            # # TODO: track where float64 is introduced
            # detections_batch = np.array(detections_batch).astype(np.float32)
            # # Reshape output
            # # [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
            # return np.reshape(detections_batch, [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

        # # Return wrapped function
        # return tf.py_func(wrapper, inputs, tf.float32)

    # def compute_output_shape(self, input_shape):
        # return (None, self.config.DETECTION_MAX_INSTANCES, 6)
        
