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
import numpy as np
import tensorflow as tf
# from collections import OrderedDict
# import keras.backend as K
# import keras.layers as KL
# import keras.initializers as KI
import keras.engine as KE
import keras.backend as KB
sys.path.append('..')
import mrcnn.utils as utils


############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    '''
    Computes IoU overlaps between two sets of boxes.in normalized coordinates
    
    boxes1 - proposals :  [batch_size,  proposal_counts, 4 (y1, x1, y2, x2)] <-- Region proposals
    boxes2 - gt_boxes  :  [batch_size, max_gt_instances, 4 (y1, x1, y2, x2)] <-- input_normlzd_gt_boxes
    
    proposal_counts : 1000 or 2000 based on training or inference
    max_gt_instances: 100
    
    returns :
    ---------
    overlaps :          [ proposal_counts, max_gt_instances] 
                        IoU of all proposal box / gt_box pairs
                        The dimensionality :
                            row:  number of non_zero proposals 
                            cols: number of non_zero gt_bboxes
    '''
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare every boxes1 against every boxes2 without loops.
    #    TF doesn't have an equivalent to np.repeat() so simulate it using tf.tile() and tf.reshape.
    
    # print('\t>>> detection_targets_graph - calculate Overlaps_graph')    
    # print('\t     overlaps_graph: shape of boxes1 before reshape: ',tf.shape(boxes1).eval())  # (?,?)
    # print('\t     overlaps_graph: shape of boxes2 before reshape: ',tf.shape(boxes1).eval())  # (?,?)
    
    # tf.expand_dims(boxes1, 1) : makes b1:[1, batch_size, proposal_count_sz, 4] 
    
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # print('\t     overlaps_graph: shape of boxes1 after reshape: ',tf.shape(b1).eval())  # (?,4)
    # print('\t     overlaps_graph: shape of boxes2 after reshape: ',tf.shape(b2).eval())  # (?,4)


    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    
    # print('     overlaps_graph: shape of b1_y1 after split: ',b2_y1.shape)  # (?,4)
    
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    print('\t     Overlaps_graph(): Shape of output overlaps', tf.shape(overlaps), overlaps.get_shape())
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    '''
    Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    -------
    proposals:          [N, 2000, (y1, x1, y2, x2)] in normalized coordinates. 
                             Might be zero padded if there are not enough proposals.
    gt_class_ids:       [MAX_GT_INSTANCES] int class IDs
    gt_boxes:           [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks:           [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns:            Target ROIs and corresponding class IDs, bounding box shifts, and masks.
    --------
    rois:               [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids:          [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas:             [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                        Class-specific bbox refinments.
    masks:              [TRAIN_ROIS_PER_IMAGE, height, width). Masks cropped to bbox
                        boundaries and resized to neural network output size.
   
    Note: Returned arrays might be zero padded if not enough target ROIs.

    ''' 
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion"),
    ]
    
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)
    print('>>> detection_targets_graph ')
    print('     propsals.shape        :',  proposals.shape, proposals.get_shape(), KB.int_shape(proposals) )
    print('     gt_boxes.shape        :',  gt_boxes.shape ,    KB.int_shape(gt_boxes)   )
    print('     gt_class_ids.shape    :',  gt_class_ids.shape, KB.int_shape(gt_class_ids))
    print('     gt_masks.shape        :',  gt_masks.shape ,    KB.int_shape(gt_masks)   )

    # Remove zero padding   
    # non_zeros returns indicies to valid bboxes, which we use to index gt_class_ids, and gt_masks
    proposals, _        = utils.trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = utils.trim_zeros_graph(gt_boxes , name="trim_gt_boxes")
    gt_class_ids        = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
    gt_masks            = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,name="trim_gt_masks")

    #------------------------------------------------------------------------------------------
    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    #------------------------------------------------------------------------------------------
    # tf.where : returns the coordinates of true elements of  the specified conditon.
    #            The coordinates are returned in a 2-D tensor where the first dimension (rows) 
    #            represents the number of true elements, and the second dimension (columns) 
    #            represents the coordinates of the true elements. 
    #            Keep in mind, the shape of the output tensor can vary depending on how many 
    #            true values there are in input. Indices are output in row-major order.
    #
    # tf.gather: Gather slices from params axis (default = 0) according to indices.
    #            indices must be an integer tensor of any dimension (usually 0-D or 1-D). 
    #            Produces an output tensor with shape:
    #                   params.shape[:axis] + indices.shape + params.shape[axis + 1:] 
    #
    # tf.squeeze: Removes dimensions of size 1 from the shape of a tensor.
    #            Given a tensor input, this operation returns a tensor of the same type with 
    #            all dimensions of size 1 removed. If you don't want to remove all size 1 
    #            dimensions, you can remove specific size 1 dimensions by specifying axis.
    #------------------------------------------------------------------------------------------
    
    crowd_ix        = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix    = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes     = tf.gather(gt_boxes, crowd_ix)
    crowd_masks     = tf.gather(gt_masks, crowd_ix, axis=2)
    gt_class_ids    = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes        = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks        = tf.gather(gt_masks, non_crowd_ix, axis=2)


    # Compute overlaps with crowd boxes [anchors, crowds]
    crowd_overlaps  = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max   = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool   = (crowd_iou_max < 0.001)


    # Compute overlaps matrix [proposals, gt_boxes] - The IoU between 
    # proposals and gt_boxes (non-crowd gt boxes, designated by classId < 0 in Coco)
    # overlaps is 
    # compute max of elements across axis 1 of overlaps tensor. 
    overlaps        = overlaps_graph(proposals, gt_boxes)
    roi_iou_max     = tf.reduce_max(overlaps, axis=1)
    # print('     overlaps.shape        :',  overlaps.shape, KB.int_shape(overlaps)   )

    ## 1. Determine indices of postive ROI propsal boxes
    #    Identify ROI proposal boxes that have an IoU >= 05 overlap with some gt_box, and store 
    #    indices into positive_indices
    positive_roi_bool     = (roi_iou_max >= 0.5)
    positive_indices      = tf.where(positive_roi_bool)[:, 0]

    ## 2. Determine indices of negative ROI proposal boxes
    #    those with < 0.5 with every GT box and are not crowds bboxes 
    # the where creates a array with shape [# of answers, 1] so we use [:, 0] after
    ## current method
    negative_indices      = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    ## new method
    # this modification will determine negative ROI proposal boxes but in addition, 
    # will suppress the zero RoIs from the indicies 
    # note that   ( negative_bool         = ~positive_roi_bool)
    # negative_nonzero_bool = tf.logical_and(~positive_roi_bool, (roi_iou_max > 0))
    # negative_nonzero_bool = tf.logical_and(negative_nonzero_bool, no_crowd_bool)
    # negative_indices2     = tf.where(negative_nonzero_bool) [:, 0]

    ## 3. Subsample positive ROIs based on ROI_POSITIVE_RATIO
    #    Aim for 33% positive (config.ROI_POSITIVE_RATIO = 0.33)
    #    Positive ROIs   33% of config.TRAIN_ROIS_PER_IMAGE ~  11
    positive_count        = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    positive_indices      = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count        = tf.shape(positive_indices)[0]
    
    ## 4. Add Negative ROIs. Add enough to maintain positive:negative ratio
    #     negative_count = int((positive_count / config.ROI_POSITIVE_RATIO) - positive_count)
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count        = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices      = tf.random_shuffle(negative_indices)[:negative_count]
    
    ## 5.   Gather selected positive and negative ROIs
    positive_rois         = tf.gather(proposals, positive_indices)
    negative_rois         = tf.gather(proposals, negative_indices)

    ## 6.   Assign positive ROIs to GT boxes.
    #      roi_gt_box_assignment shows for each positive overlap, which class has the maximum overlap
    positive_overlaps     = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
    roi_gt_boxes          = tf.gather(gt_boxes    , roi_gt_box_assignment)
    roi_gt_class_ids      = tf.gather(gt_class_ids, roi_gt_box_assignment)
    # print('     shape of positive overlaps is :', positive_overlaps.get_shape())

    ## 7.   Compute bbox delta 
    # calculate refinement (difference b/w positive rois and gt_boxes) for positive ROIs
    deltas  = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    ## 8.  prepare gt_masks 
    #      transpose gt_masks from [h, w, N] to [N, height, width] and add 4th dim at end [N, height, width, 1]
    #      Pick the right mask for each ROI
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    
    if config.USE_MINI_MASK:
        # Transform ROI corrdinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks   = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), 
                                       boxes,
                                       box_ids,
                                       config.MASK_SHAPE)
                                     
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois             = tf.concat([positive_rois, negative_rois], axis=0)
    N                = tf.shape(negative_rois)[0]
    P                = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois             = tf.pad(rois            , [(0, P), (0, 0)])
    
    roi_gt_boxes     = tf.pad(roi_gt_boxes    , [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas           = tf.pad(deltas          , [(0, N + P), (0, 0)])
    masks            = tf.pad(masks           , [[0, N + P], (0, 0), (0, 0)])
    
    # print(' roi_gt_boxes :  ' , tf.shape(roi_gt_boxes) )
    # print(' P:  ' , P,  ' N :    ', N)   
    # print('     roi.shape             :',  rois.shape            , tf.shape(rois))
    # print('     roi_gt_class_ids.shape:',  roi_gt_class_ids.shape, tf.shape(roi_gt_class_ids))
    # print('     deltas.shape          :',  deltas.shape          , tf.shape(deltas))
    # print('     masks.shape           :',  masks.shape           , tf.shape(masks))
    # print('     roi_gt_boxes.shape    :',  roi_gt_boxes.shape    , tf.shape(roi_gt_boxes))
    
    return rois, roi_gt_class_ids,  deltas, masks, roi_gt_boxes


class DetectionTargetLayer(KE.Layer):
    '''
    Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    -------    
    proposals:          [batch, N, (y1, x1, y2, x2)] in normalized coordinates.
                          Proposal bboxes generated by RPN
                          Might be zero padded if there are not enough proposals.
    gt_class_ids:       [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes:           [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks:           [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: 
    -------    
    Target ROIs and corresponding class IDs, bounding box shifts, and masks.
    
    rois:               [batch, TRAIN_ROIS_PER_IMAGE, 4 {y1, x1, y2, x2}] 
                        Normalized coordinates
    
    target_class_ids:   [batch, TRAIN_ROIS_PER_IMAGE, 1]. Integer class IDs.
    
    target_deltas:      [batch, TRAIN_ROIS_PER_IMAGE, 4 {dy, dx, log(dh), log(dw)}]
                        Class-specific bbox refinments.
    
    target_mask:        [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                        Masks cropped to bbox boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    '''

    def __init__(self, config, **kwargs):
        # super(DetectionTargetLayer, self).__init__(**kwargs)
        super().__init__(**kwargs)
        print('\n>>> Detection Target Layer ')
        self.config = config

    def call(self, inputs):
        print('    Detection Target Layer : call() ', type(inputs), len(inputs))
        print('     proposals.shape    :',  inputs[0].shape, inputs[0].get_shape(), KB.int_shape(inputs[0]) )
        print('     gt_class_ids.shape :',  inputs[1].shape, inputs[1].get_shape(), KB.int_shape(inputs[1]) ) 
        print('     gt_bboxes.shape    :',  inputs[2].shape, inputs[2].get_shape(), KB.int_shape(inputs[2]) )
        print('     gt_masks.shape     :',  inputs[3].shape, inputs[3].get_shape(), KB.int_shape(inputs[3]) ) 
        proposals    = inputs[0]    # target_rois           --  proposals generated by the RPN (or artificially generated proposals)
        gt_class_ids = inputs[1]    # input_gt_class_ids 
        gt_boxes     = inputs[2]    # input_normlzd_gt_boxes
        gt_masks     = inputs[3]    # input_gt_masks

        # Slice the batch and run a graph for each slice    
        # TODO: Rename target_bbox to target_deltas for clarity
        # detection_target_graph() returns:
        #         rois,    roi_gt_class_ids,  deltas,         masks,       roi_gt_boxes
        
        names = ["output_rois", "target_class_ids", "target_bbox_deltas", "target_mask", "roi_gt_boxes"]
        
        outputs = utils.batch_slice([proposals, gt_class_ids, gt_boxes, gt_masks],             # inputs 
                                    lambda w, x, y, z: detection_targets_graph(w, x, y, z, self.config),        # batch function 
                                    self.config.IMAGES_PER_GPU,                                                 # batch_size, name
                                    names=names)                  
                   
        print('\n    Detection Target Layer : return ', type(outputs) , len(outputs))                    
        for i,out in enumerate(outputs):
            print('     output {}  shape {}  type {} '.format(i, out.shape, type(out)))    

        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),                                                        # rois
            (None, 1),                                                                                          # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),                                                        # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0], self.config.MASK_SHAPE[1]),     # masks
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4)                                                         # roi_gt_bboxes

        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None, None]


