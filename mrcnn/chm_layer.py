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
from scipy.stats import  multivariate_normal
# import scipy.misc
import tensorflow as tf
# import keras
import keras.backend as KB
import keras.layers as KL
import keras.engine as KE
sys.path.append('..')
import mrcnn.utils as utils
import tensorflow.contrib.util as tfc
import pprint

     
    
def build_predictions_tf(mrcnn_class, norm_output_rois, config):
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES
    h, w            = config.IMAGE_SHAPE[:2]
    # num_rois        = config.TRAIN_ROIS_PER_IMAGE
    num_cols        = 6
    num_rois        = KB.int_shape(norm_output_rois)[1]
    output_rois = norm_output_rois * np.array([h,w,h,w])   

    print()
    print('  > BUILD_PREDICTIONS_TF()')
    print('    num_rois          : ', num_rois )
    print('    mrcnn_class shape : ', KB.shape(mrcnn_class), KB.int_shape(mrcnn_class))
    # print('    mrcnn_bbox.shape  : ', KB.shape(mrcnn_bbox) , KB.int_shape(mrcnn_bbox), mrcnn_bbox.shape )
    print('    output_rois.shape : ', KB.shape(output_rois), KB.int_shape(output_rois))
    #---------------------------------------------------------------------------
    # use the argmaxof each row to determine the dominating (predicted) class
    #---------------------------------------------------------------------------
    pred_classes     = tf.to_int32(tf.argmax( mrcnn_class,axis=-1))
    pred_classes_exp = tf.to_float(tf.expand_dims(pred_classes ,axis=-1))
    pred_scores      = tf.reduce_max(mrcnn_class ,axis=-1, keepdims=True)   # (32,)
    
    # np.set_printoptions(linewidth=100, precision=4)
    # print('    pred_classes with highest scores:', pred_classes.get_shape() )
    # print('    pred_ scores:', pred_scores.get_shape())
    
    print('    pred_classes     : ', pred_classes.shape)
    print('    pred_classes_exp : ', pred_classes_exp.shape)
    print('    pred_scores      : ', pred_scores.shape)
    
    #-----------------------------------------------------------------------------------
    # This part is used if we want to gather bbox coordinates from mrcnn_bbox 
    #  Currently we are gathering bbox coordinates form output_roi so we dont need this
    #-----------------------------------------------------------------------------------
    # bbox_selected    = tf.zeros_like(norm_output_rois)
    # print('    bbox_selected    : ', bbox_selected.shape)
    # gather_boxes    = tf.stack([batch_grid, roi_grid, pred_classes, ], axis = -1)
    # print('-- gather_boxes  ----')
    # print('gather_boxes inds', type(gather_boxes), 'shape',tf.shape(gather_boxes).eval())
    # print(gather_boxes.eval())
    # bbox_selected   = tf.gather_nd(mrcnn_bbox, gather_boxes)
    # print('    bbox_selected shape : ', bbox_selected.get_shape())
    # print(bbox_selected[0].eval())    
    #-----------------------------------------------------------------------------------
    
    ## moved pred_scores to end -- 30-04-2018
    # pred_array  = tf.concat([bbox_idx , output_rois, pred_classes_exp , pred_scores], axis=-1)
    pred_array  = tf.concat([ output_rois, pred_classes_exp , pred_scores], axis=-1)

    #---------------------------------------------------------------------------
    # Build a meshgrid for image id and bbox to use in gathering of bbox information 
    #---------------------------------------------------------------------------
    batch_grid, roi_grid = tf.meshgrid( tf.range(batch_size, dtype=tf.int32),
                                        tf.range(num_rois, dtype=tf.int32), indexing = 'ij' )
    # bbox_idx             = tf.to_float(tf.expand_dims(roi_grid , axis = -1))    
    print('    batch_grid       : ', batch_grid.shape)
    print('    roi_grid         : ', roi_grid.shape)


    scatter_ind          = tf.stack([batch_grid , pred_classes, roi_grid],axis = -1)
    pred_scatt = tf.scatter_nd(scatter_ind, pred_array, [batch_size, num_classes, num_rois, pred_array.shape[-1]])
    # print('scatter_ind', type(scatter_ind), 'shape',tf.shape(scatter_ind).eval())
    # print('    pred_scatter shape is ', pred_scatt.get_shape(), pred_scatt)
    
    #------------------------------------------------------------------------------------
    ## sort pred_scatter in each class dimension based on sequence number (last column)
    #------------------------------------------------------------------------------------
    _, sort_inds = tf.nn.top_k(pred_scatt[:,:,:,-1], k=pred_scatt.shape[2])

    # build indexes to gather rows from pred_scatter based on sort order    
    class_grid, batch_grid, roi_grid = tf.meshgrid(tf.range(num_classes),tf.range(batch_size), tf.range(num_rois))
    roi_grid_exp = tf.to_float(tf.expand_dims(roi_grid, axis = -1))
    
    gather_inds  = tf.stack([batch_grid , class_grid, sort_inds],axis = -1)
    pred_tensor  = tf.gather_nd(pred_scatt, gather_inds, name = 'pred_tensor')
    
    # print('    class_grid  ', type(class_grid) , 'shape', class_grid.get_shape())
    # print('    batch_grid  ', type(batch_grid) , 'shape', batch_grid.get_shape())
    # print('    roi_grid shape', roi_grid.get_shape(), 'roi_grid_exp shape ', roi_grid_exp.get_shape())
    # print('    gather_inds ', type(gather_inds), 'shape', gather_inds.get_shape())
    # print('    pred_tensor (gathered)  : ', pred_tensor.get_shape())

    # append an index to the end of each row --- commented out 30-04-2018
    # pred_tensor  = tf.concat([pred_tensor, roi_grid_exp], axis = -1)
    
    # print('    final pred_tensor shape  : ', pred_tensor.get_shape())
    # print('    complete')

    return  pred_tensor
    
    
    
def build_ground_truth_tf(gt_class_ids, gt_bboxes, num_bboxes, config):
    # // pass model to TensorBuilder
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES
    h, w            = config.IMAGE_SHAPE[:2]
    num_cols        = 7
    
    # num of bounding boxes is determined by bbox_list.shape[1] instead of config.DETECTION_MAX_INSTANCES
    # use of this routine for both input_gt_boxes, and target_gt_deltas
    if num_bboxes == config.DETECTION_MAX_INSTANCES:
        tensor_name = "gt_tensor"
    else:
        tensor_name = "gt_deltas"
        
    print('\n')
    print('  > BUILD_GROUND TRUTH_TF()' )
    print('    gt_class_ids shape : ', gt_class_ids.get_shape())
    print('    gt_bboxes.shape    : ', gt_bboxes.get_shape() )

    #---------------------------------------------------------------------------
    # use the argmaxof each row to determine the dominating (predicted) class
    # mask identifies class_ids > 0 
    #---------------------------------------------------------------------------
    gt_classes_exp = tf.to_float(tf.expand_dims(gt_class_ids ,axis=-1))
    print('    gt_classes_exp shape ', gt_classes_exp.get_shape() )

    ones = tf.ones_like(gt_class_ids)
    zeros= tf.zeros_like(gt_class_ids)
    mask = tf.greater(gt_class_ids , 0)

    gt_scores     = tf.where(mask, ones, zeros)
    # pred_scores      = tf.reduce_max(mrcnn_class ,axis=-1, keep_dims=True)   # (32,)
    gt_scores_exp = tf.to_float(KB.expand_dims(gt_scores, axis=-1))
    print('    gt_scores_exp shape ', gt_scores_exp.get_shape())

    #---------------------------------------------------------------------------
    # create meshgrid to do something 
    #---------------------------------------------------------------------------
    batch_grid, bbox_grid = tf.meshgrid( tf.range(batch_size, dtype=tf.int32), 
                                         tf.range(num_bboxes, dtype=tf.int32), indexing = 'ij' )
    
    # sequence id is used to preserve the order of rois as passed to this routine
    sequence = gt_scores * (bbox_grid[...,::-1] + 1) 
    sequence = tf.to_float(tf.expand_dims(sequence, axis = -1))   

    # print('    batch_grid shape  ', batch_grid.get_shape())
    # print(batch_grid.eval(session=sess))
    # print('    bbox_grid  shape  ', bbox_grid.get_shape())
    # print(bbox_grid.eval(session=sess))
    # print('    sequence shape    ', sequence.get_shape())
    # print(sequence.eval(session=sess)) 

    # obsolete replaced with sequence 12-05-2018    
    # bbox_idx_zeros  = tf.zeros_like(bbox_grid)
    # bbox_idx        = tf.where(mask, bbox_grid , bbox_idx_zeros)
    # bbox_idx        = tf.to_float(tf.expand_dims(bbox_idx, axis = -1))    
    
    # 12-05-2018 (added sequecne)    gt_array        = tf.concat([gt_bboxes, gt_classes_exp, gt_scores_exp], axis=2)
    gt_array        = tf.concat([gt_bboxes, gt_classes_exp, gt_scores_exp, sequence ], axis=2)
    print('    gt_array shape :', gt_array.shape, gt_array.get_shape())

    #------------------------------------------------------------------------------
    # Create indicies to scatter rois out to multi-dim tensor by image id and class
    # resulting tensor is batch size x num_classes x num_bboxes x 7 (num columns)
    #------------------------------------------------------------------------------
    scatter_ind = tf.stack([batch_grid , gt_class_ids, bbox_grid],axis = -1)
    # obsolete 12-05-2018 
    # gt_scatter = tf.scatter_nd(scatter_ind, gt_array, [batch_size, num_classes, num_bboxes, num_cols] )
    gt_scatter = tf.scatter_nd(scatter_ind, gt_array, [batch_size, num_classes, num_bboxes, num_cols ])

    # print('    scatter_ind shape ', scatter_ind.get_shape())
    # print('    gt_scatter shape ', gt_scatter.get_shape())
    
    #-------------------------------------------------------------------------------
    ## sort in each class dimension based on on sequence number (last column)
    # scatter_nd places bboxs in a sparse fashion --- this sort is to place all bboxes
    # at the top of the class bbox array
    #-------------------------------------------------------------------------------

    _ , sort_inds = tf.nn.top_k(tf.abs(gt_scatter[:,:,:,-1]), k=gt_scatter.shape[2])

    # build indexes to gather rows from pred_scatter based on sort order 
    class_grid, batch_grid, bbox_grid = tf.meshgrid(tf.range(num_classes),tf.range(batch_size), tf.range(num_bboxes))
    bbox_grid_exp = tf.to_float(tf.expand_dims(bbox_grid, axis = -1))
 
    gather_inds = tf.stack([batch_grid , class_grid, sort_inds],axis = -1)
    gt_result   = tf.gather_nd(gt_scatter[...,:-1], gather_inds, name  = tensor_name)
    # gt_result   = gt_result[...,:-1]
    # print('    gather_inds shape      : ', gather_inds.get_shape())
    
    # append an index to the end of each row --- commented out 30-04-2018
    # gt_tensor   = tf.concat([gt_tensor, bbox_grid_exp], axis = -1)

    # count based on pred score > 0 (changed from index 0 to -1 on 30-04-2018)   
    # moved outside to call() 12-05-2018
    # gt_cls_cnt  = tf.count_nonzero(gt_result[:,:,:,-1],axis = -1, name = 'gt_cls_count')

    
    print('    ',tensor_name, 'final shape  : ', gt_result.get_shape())

    return  gt_result

    

    
def build_heatmap(in_tensor, config, names = None):

    num_detections  = config.DETECTION_MAX_INSTANCES
    img_h, img_w    = config.IMAGE_SHAPE[:2]
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES  
    print('\n ')
    print('  > NEW build_heatmap() for ', names )
    print('    orignal in_tensor shape : ', in_tensor.shape)       
    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES
    rois_per_image  = (in_tensor.shape)[2] 
    # strt_cls        = 0 if rois_per_image == 32 else 1
    print('    num of bboxes per class is : ', rois_per_image )

    #-----------------------------------------------------------------------------    
    ## Stack non_zero bboxes from in_tensor into pt2_dense 
    #-----------------------------------------------------------------------------
    # pt2_ind shape is [?, 3]. 
    #   pt2_ind[0] corresponds to image_index 
    #   pt2_ind[1] corresponds to class_index 
    #   pt2_ind[2] corresponds to roi row_index 
    # pt2_dense shape is [?, 6]
    #    pt2_dense[0] is image index
    #    pt2_dense[1:4]  roi cooridnaytes 
    #    pt2_dense[5]    is class id 
    #-----------------------------------------------------------------------------
    pt2_sum = tf.reduce_sum(tf.abs(in_tensor[:,:,:,:-2]), axis=-1)
    print('    pt2_sum shape ',pt2_sum.shape)
    # print(pt2_sum[0].eval())
    pt2_ind = tf.where(pt2_sum > 0)

    ## replaced the two operations below with the one above - 15-05-2018
    # pt2_mask = tf.greater(pt2_sum , 0)
    # pt2_ind  = tf.where(pt2_mask)
    # print(' pt2_mask shape ', pt2_mask.get_shape())
    # print(pt2_mask.eval())
    # print('    pt2_ind shape ', pt2_ind.get_shape())
    # print(pt2_ind.eval())

    pt2_dense = tf.gather_nd( in_tensor, pt2_ind)
    print('    dense shape ',pt2_dense.get_shape())
    
    #-----------------------------------------------------------------------------
    ## Build mesh-grid to hold pixel coordinates  
    #-----------------------------------------------------------------------------
    X = tf.range(img_w, dtype=tf.int32)
    Y = tf.range(img_h, dtype=tf.int32)
    X, Y = tf.meshgrid(X, Y)
    
    # duplicate (repeat) X and Y into a  batch_size x rois_per_image tensor 
    print('    X/Y shapes :',  X.get_shape(), Y.get_shape())
    ones = tf.ones([tf.shape(pt2_dense)[0] , 1, 1], dtype = tf.int32)
    rep_X = ones * X
    rep_Y = ones * Y 
    print('    Ones:    ', ones.shape)                
    print('    ones_exp * X', ones.shape, '*', X.shape, '= ',rep_X.shape)
    print('    ones_exp * Y', ones.shape, '*', Y.shape, '= ',rep_Y.shape)

    # # stack the X and Y grids 
    bef_pos = tf.to_float(tf.stack([rep_X,rep_Y], axis = -1))
    print('    before transpse ', bef_pos.get_shape())
    pos_grid = tf.transpose(bef_pos,[1,2,0,3])
    print('    after transpose ', pos_grid.get_shape())    
    
    #-----------------------------------------------------------------------------
    ##  Build mean and convariance tensors for Multivariate Normal Distribution 
    #-----------------------------------------------------------------------------
    width  = pt2_dense[:,3] - pt2_dense[:,1]      # x2 - x1
    height = pt2_dense[:,2] - pt2_dense[:,0]
    cx     = pt2_dense[:,1] + ( width  / 2.0)
    cy     = pt2_dense[:,0] + ( height / 2.0)
    means  = tf.stack((cx,cy),axis = -1)
    covar  = tf.stack((width * 0.5 , height * 0.5), axis = -1)
    covar  = tf.sqrt(covar)

    tfd = tf.contrib.distributions
    mvn = tfd.MultivariateNormalDiag( loc  = means,  scale_diag = covar)
    prob_grid = mvn.prob(pos_grid)
    print('     Prob_grid shape before tanspose: ',prob_grid.get_shape())
    prob_grid = tf.transpose(prob_grid,[2,0,1])
    print('     Prob_grid shape after tanspose: ',prob_grid.get_shape())    
    print('    >> input to MVN.PROB: pos_grid (meshgrid) shape: ', pos_grid.get_shape())
    print('    << output probabilities shape:' , prob_grid.get_shape())
    
    #--------------------------------------------------------------------------------
    ## IMPORTANT: kill distributions of NaN boxes (resulting from bboxes with height/width of zero
    ## which cause singular sigma cov matrices
    #--------------------------------------------------------------------------------
    prob_grid = tf.where(tf.is_nan(prob_grid),  tf.zeros_like(prob_grid), prob_grid)

    
    ## scatter out the probability distributions based on class --------------------------
    print('\n    Scatter out the probability distributions based on class --------------') 
    gauss_scatt   = tf.scatter_nd(pt2_ind, prob_grid, [batch_size, num_classes, rois_per_image, img_w, img_h])
    print('    pt2_ind shape   : ', pt2_ind.shape)  
    print('    prob_grid shape : ', prob_grid.shape)  
    print('    gauss_scatt     : ', gauss_scatt.shape)   # batch_sz , num_classes, num_rois, image_h, image_w
    
    ## heatmap: sum gauss_scattered based on class ---------------------------------------
    print('\n    Reduce sum based on class ---------------------------------------------')         
    gauss_sum = tf.reduce_sum(gauss_scatt, axis=2, name='pred_heatmap2')
    gauss_sum = tf.where(gauss_sum > 1e-12, gauss_sum, tf.zeros_like(gauss_sum))
    print('    gaussian_sum shape     : ', gauss_sum.get_shape(), 'Keras tensor ', KB.is_keras_tensor(gauss_sum) )  
    # reshape to [img, class, height, width] ---> [img, height, width, class]
    gauss_sum = tf.transpose(gauss_sum,[0,2,3,1], name = names[0])   
    print('    gaussian sum type/name : ', type(gauss_sum), gauss_sum.name, names[0])
    print('    gaussian_sum shape     : ', gauss_sum.get_shape(), 'Keras tensor ', KB.is_keras_tensor(gauss_sum) )      

    ## heatmap: L2 normalization  -----------------------------------------------------------------
    print('\n    L2 normalization ------------------------------------------------------')   
    heatmap_shape=KB.shape(gauss_sum)
    print('    gauss-sum.shape:', gauss_sum.shape, 'tf.shape :', tf.shape(gauss_sum))
   
    gauss_flatten = KB.reshape(gauss_sum, (heatmap_shape[0], -1, heatmap_shape[-1]) )  # reshape to image, class
    output_norm   = KB.l2_normalize(gauss_flatten, axis = 1)    
    gauss_norm    = KB.identity(KB.reshape(output_norm,  heatmap_shape ) , name = names[0]+'_norm')   

    print('    gauss_flatten    : ', KB.int_shape(gauss_flatten) , gauss_flatten.get_shape(),' Keras tensor ', KB.is_keras_tensor(gauss_flatten) )
    print('    gauss_norm1      : ', KB.int_shape(output_norm)   , output_norm.get_shape(),' Keras tensor ', KB.is_keras_tensor(output_norm) )
    print('    gauss_norm final : ', KB.int_shape(gauss_norm)    , gauss_norm.get_shape(),' Keras tensor ', KB.is_keras_tensor(gauss_norm) )
   
    ##--------------------------------------------------------------------------------------------
    ## generate score based on gaussian using bouding box masks 
    ## NOTE: Score is generated on NON-NORMALIZED gaussian distributions
    ##       If want to do this on normalized, we need to apply normalization to gauss_scatt first
    ##--------------------------------------------------------------------------------------------
    # flatten guassian scattered and input_tensor, and pass on to build_bbox_score routine 
    in_tensor_flattened  = tf.reshape(in_tensor, [-1,6])
    bboxes = tf.to_int32(tf.round(in_tensor_flattened[...,0:4]))
    print('    in_tensor_flattened is ', in_tensor_flattened.shape)
    print('    boxes shape          ', bboxes.shape)

    # DONT NEED THIS - was put there to try to avoid computing sum/area for zero bboxes.
    # kept as reference for future generations .....
    # bbox_sum = tf.reduce_max(in_tensor[...,0:3], axis = -1, name  = 'bbox_sum')
    # print(' bbox sum shape: ', bbox_sum.shape)

    gauss_scatt_shape   = KB.int_shape(gauss_scatt)
    gauss_scatt_reshape = KB.reshape(gauss_scatt, (-1, gauss_scatt_shape[-2], gauss_scatt_shape[-1]))
    print('    gaussian scatter shape : ', gauss_scatt_shape)
    print('    gaussian scatter reshaped : ', gauss_scatt_reshape.shape)
    # ones_map = tf.ones([384,128,128])   
    scores = tf.map_fn(build_mask_routine, [gauss_scatt_reshape, bboxes], dtype=tf.float32)
    
    new_shape = tf.shape(in_tensor)+ [0,0,0,tf.shape(scores)[-1]]        
    gaussian_bbox_scores = tf.concat([in_tensor_flattened, scores], axis = -1)
    print('    Scatter Flattened shape : ', in_tensor_flattened.shape)
    print('    Scores shape :            ', scores.shape)
    print('    gaussian_boxes_scores initial shape: ', gaussian_bbox_scores.shape)    
    gaussian_bbox_scores = tf.reshape(gaussian_bbox_scores, new_shape, name = names[0]+'_scores')

    ##--------------------------------------------------------------------------------------------
    ## Normalize computed score above, and add it to the heatmap_score tensor as last column
    ##--------------------------------------------------------------------------------------------
    scr = gaussian_bbox_scores[...,-2]/gaussian_bbox_scores[...,-1]
    scr = tf.where(tf.is_nan(scr),  tf.zeros_like(scr), scr)       
    scr_norm = tf.nn.l2_normalize(scr, axis = -1)
    scr_norm = tf.expand_dims(scr_norm, axis = -1)
    gaussian_bbox_scores = tf.concat([gaussian_bbox_scores, scr_norm], axis = -1)
    
    print('    gaussian_bbox_scores final shape   : ', gaussian_bbox_scores.shape)
    print('    complete')

    return  gauss_norm, gaussian_bbox_scores    # [gauss_sum, gauss_scatt, means, covar]
    
    
    
    
def build_mask_routine(input_list):
    '''
    Inputs:
    -----------
        heatmap_tensor :    [ image height, image width ]
        input_row      :    [y1, x1, y2, x2] in absolute (non-normalized) scale

    Returns
    -----------
        gaussian_sum :      sum of gaussian heatmap vlaues over the area covered by the bounding box
        bbox_area    :      bounding box area (in pixels)
    '''
    heatmap_tensor, input_row = input_list
    with tf.variable_scope('mask_routine'):
        y_extent     = tf.range(input_row[0], input_row[2])
        x_extent     = tf.range(input_row[1], input_row[3])
        Y,X          = tf.meshgrid(y_extent, x_extent)
        bbox_mask    = tf.stack([Y,X],axis=2)        
        mask_indices = tf.reshape(bbox_mask,[-1,2])
        mask_indices = tf.to_int32(mask_indices)
        mask_size    = tf.shape(mask_indices)[0]
        mask_updates = tf.ones([mask_size], dtype = tf.float32)    
        mask         = tf.scatter_nd(mask_indices, mask_updates, tf.shape(heatmap_tensor))
        mask_applied = tf.multiply(heatmap_tensor, mask, name = 'mask_applied')
        bbox_area    = tf.to_float((input_row[2]-input_row[0]) * (input_row[3]-input_row[1]))
        gaussian_sum = tf.reduce_sum(mask_applied)
    return tf.stack([gaussian_sum, bbox_area], axis = -1)

    
    
    
    
## still using the old method!
def build_heatmap_inference(in_tensor, config, names = None):
    '''
    detections :   [N, 100, 6 {y1, x1, y2, x2}]
    '''
    # rois_per_image  = 32    
    num_detections  = config.DETECTION_MAX_INSTANCES
    h, w            = config.IMAGE_SHAPE[:2]
    img_h, img_w    = config.IMAGE_SHAPE[:2]
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES  
    print('\n')
    print('  > BUILD_HEATMAP_INFERENCE() for ' , names)
    
    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES
    
    print('    in_tensor shape : ', in_tensor.shape)   
    # in_tensor = in_tensor[:,:,:,2:7]
    # print('    modified in_tensor shape : ', in_tensor.get_shape())
    
    rois_per_image  = tf.to_int32(in_tensor.shape[2])
    strt_cls        = 0 if rois_per_image == 32 else 1
    print('    num of bboxes per class is : ', rois_per_image)

    ## Build mesh-grid to hold pixel coordinates ----------------------------------
    X = tf.range(img_w, dtype=tf.int32)
    Y = tf.range(img_h, dtype=tf.int32)
    X, Y = tf.meshgrid(X, Y)
    # print('    X/Y shapes :',  X.get_shape(), Y.get_shape())
    # print('    X : \n',X.eval())
    # print('    Y : \n',Y.eval())

    ## repeat X and Y  batch_size x rois_per_image times
    ones = tf.ones([batch_size, rois_per_image,1, 1], dtype = tf.int32)
    rep_X = ones * X
    rep_Y = ones * Y 
    # print('    Ones: ',ones.shape)                
    # print(' ones_exp * X', ones.shape, '*', X.shape, '= ',rep_X.shape)
    # print(' ones_exp * Y', ones.shape, '*', Y.shape, '= ',rep_Y.shape)

    # # stack the X and Y grids 
    bef_pos = tf.to_float(tf.stack([rep_X,rep_Y], axis = -1))
    # print(' before transpse ', bef_pos.get_shape())
    pos_grid = tf.transpose(bef_pos,[2,3,0,1,4])
    print('    after transpose ', pos_grid.get_shape())    

    # pt2_reshape = tf.reshape( in_tensor , [batch_size, num_classes * rois_per_image ,8])
    # print('    pt2_reshape shape is : ', pt2_reshape.get_shape())
    # print(pt2_reshape[0].eval())
    # print(pt2_reshape[1].eval())
    # print(pt2_reshape[2].eval())

    ## Stack non_zero bboxes from in_tensor into pt2_dense --------------------------
    pt2_sum = tf.reduce_sum(tf.abs(in_tensor[:,:,:,:-1]), axis=-1)
    print('    pt2_sum shape ',pt2_sum.shape)
    # print(pt2_sum[0].eval())

    pt2_mask = tf.greater(pt2_sum , 0)
    # print(' pt2_mask shape ', pt2_mask.get_shape())
    # print(pt2_mask.eval())

    pt2_ind  = tf.where(pt2_mask)
    # print('    pt2_ind shape ', pt2_ind.get_shape())
    # print(pt2_ind.eval())
    # pt2_ind_float  =  tf.to_float(pt2_ind[:,0:1])

    pt2_dense = tf.gather_nd( in_tensor, pt2_ind)
    # print('    dense shape ',pt2_dense.get_shape())
    # print(dense.eval())

    pt2_dense = tf.concat([tf.to_float(pt2_ind[:,0:1]), pt2_dense],axis=1)
    print('    dense shape ',pt2_dense.get_shape())
    # print(dense.eval())

    # print(dense[1].eval())
    # print(dense[2].eval())
    # print(dense[3].eval())
    stacked_list = tf.dynamic_partition(pt2_dense, tf.to_int32(pt2_ind[:,0]),num_partitions = batch_size )
    # print(len(dyn_part))      

    ##  Build Stacked output from dynamically partitioned lists ----------------------
    print('    Build Stacked output from dynamically partitioned lists --------------')  

    stacked_output=[]
    for img, item  in enumerate(stacked_list) :
        # rois_in_image, cols  = tf.shape(stacked_list[img]).eval()
  
        # print('   img ', img, ' stacked_list[img] ', tf.shape(item).eval() ) 
        rois_in_image  = tf.shape(item)[0]
        #     print(stacked_list[img].eval())            
        pad_item =  tf.pad(item,[[0, rois_per_image - rois_in_image ],[0,0]])
        stacked_output.append(pad_item)
        # print()
        # print('    ===> list item #', img)     
        # print('         stacked_list[img] shape: ',rois_in_image)
        # print('         tensor_list item pos padding :', tf.shape(pad_item))
        #     print(stacked_list[img].eval())

    print()    
    stacked_tensor = tf.stack(stacked_output)
    # print('    stacked_tensor shape : ', tf.shape(stacked_tensor), stacked_tensor.shape, stacked_tensor.get_shape())
    # # print('   -- Stacked output contents --------------')    
    # # for img, item  in enumerate(stacked_output) :
        # # print('\n   ===> list item #', img)       
        # print('   img ', img, ' stacked_list[img] ', tf.shape(item).eval() ) 
        # print('   img ', img, ' stacked_list[img] ', tf.shape(item).eval()[0] ) 


    width  = stacked_tensor[:,:,4] - stacked_tensor[:,:,2]
    height = stacked_tensor[:,:,3] - stacked_tensor[:,:,1]
    cx     = stacked_tensor[:,:,2] + ( width  / 2.0)
    cy     = stacked_tensor[:,:,1] + ( height / 2.0)
    means  = tf.stack((cx,cy),axis = -1)
    covar  = tf.stack((width * 0.5 , height * 0.5), axis = -1)
    covar  = tf.sqrt(covar)
    # print(means.eval())
    # print(covar.eval())

    # print('width shape ',width.get_shape()) 
    # print(mns.eval())
    tfd = tf.contrib.distributions
    mvn = tfd.MultivariateNormalDiag( loc  = means,  scale_diag = covar)
    prob_grid  = mvn.prob(pos_grid)
    trans_grid = tf.transpose(prob_grid,[2,3,0,1])

    # print('    means shape ', means.get_shape(),' covar shape ', covar.get_shape())
    # print('    from MVN    : mns shape      : ', means.shape, means.get_shape())
    # print('    from MVN    : cov shape      : ', covar.shape, covar.get_shape())
    # print('    from MVN    : mean shape     : ', mvn.mean().get_shape(), '\t stddev shape', mvn.stddev().get_shape())
    # print('    from MVN    : mvn.batch_shape: ', mvn.batch_shape , '\t mvn.event_shape ',  mvn.event_shape)
    # print('    from Linear : op shape       : ', mvn.scale.shape, ' Linear Op batch shape ',mvn.scale.batch_shape)
    # print('    from Linear : op Range Dim   : ', mvn.scale.range_dimension)
    # print('    from Linear : op Domain Dim  : ', mvn.scale.domain_dimension) 
    print('    >> input to MVN.PROB: pos_grid (meshgrid) shape: ', pos_grid.get_shape())
    print('    << output probabilities shape:' , prob_grid.get_shape())
    # print(prob_grid.eval())
    
    #--------------------------------------------------------------------------------
    # kill distributions of NaN boxes (resulting from bboxes with height/width of zero
    # which cause singular sigma cov matrices
    #--------------------------------------------------------------------------------
    gauss_grid = tf.where(tf.is_nan(trans_grid),  tf.zeros_like(trans_grid), trans_grid)


    ## scatter out the probability distributions based on class ---------------------------
    print('\n    Scatter out the probability distributions based on class --------------')     
    class_inds      = tf.to_int32(stacked_tensor[:,:,-1])
    batch_grid, roi_grid = tf.meshgrid( tf.range(batch_size, dtype=tf.int32), tf.range(rois_per_image, dtype=tf.int32),
                                        indexing = 'ij' )
    scatter_classes = tf.stack([batch_grid, class_inds, roi_grid ],axis = -1)
    gauss_scatt     = tf.scatter_nd(scatter_classes, gauss_grid, [batch_size, num_classes, rois_per_image, img_w, img_h])

    print('    gaussian_grid      : ', gauss_grid.shape)    
    print('    class shape        : ', class_inds.shape)
    print('    roi_grid shape     : ', roi_grid.get_shape() )
    print('    batch_grid shape   : ', batch_grid.get_shape())
    print('    scatter_classes    : ', scatter_classes.get_shape())
    print('    gaussian scattered : ', gauss_scatt.shape)   
    
    ## sum based on class -----------------------------------------------------------------
    print('\n    Reduce sum based on class ---------------------------------------------')         
    gauss_sum = tf.reduce_sum(gauss_scatt, axis=2, name='pred_heatmap')
    gauss_sum = tf.where(gauss_sum > 1e-6, gauss_sum,tf.zeros_like(gauss_sum))
    gauss_sum = tf.transpose(gauss_sum,[0,2,3,1], name = names[0])
    print('    gaussian sum type/name : ', type(gauss_sum), gauss_sum.name, names[0])
    print('    gaussian_sum shape     : ', gauss_sum.get_shape(), 'Keras tensor ', KB.is_keras_tensor(gauss_sum) )    

    ## L2 normalization  -----------------------------------------------------------------
    # as done below, the heatmap of image/class is normalized seaprately 
    # if we want to do this differently, the reshape has to be changed 
    print('\n    L2 normalization ------------------------------------------------------')   
    heatmap_shape=KB.shape(gauss_sum)    # num_batches x width x height x num_classes
    gauss_flatten = KB.reshape(gauss_sum, (heatmap_shape[0], -1, heatmap_shape[-1]) )
    output_norm   = KB.l2_normalize(gauss_flatten, axis = 1)    
    gauss_norm    = KB.identity(KB.reshape(output_norm,  heatmap_shape ) , name = names[0]+'_norm')   

    print('   gauss_flatten    : ', KB.int_shape(gauss_flatten) , gauss_flatten.get_shape(),' Keras tensor ', KB.is_keras_tensor(gauss_flatten) )
    print('   gauss_norm1      : ', KB.int_shape(output_norm)   , output_norm.get_shape(),' Keras tensor ', KB.is_keras_tensor(output_norm) )
    print('   gauss_norm final : ', KB.int_shape(gauss_norm)    , gauss_norm.get_shape(),' Keras tensor ', KB.is_keras_tensor(gauss_norm) )
    print('    complete')
    
    ## Calculate Sum of Gaussian distribution over the bounding boxes -------------------
    reshape_tensor  = tf.reshape(in_tensor, [-1,6])
    # num_imgs, num_classes, num_rois, cols = KB.int_shape(input_tensor)
    print('input_shape is ', in_tensor.shape) 
    # print(' Or: ', num_imgs, num_classes, num_rois, cols)
    num_boxes, num_cols = KB.int_shape(reshape_tensor)
    print('reshape_tensor is ', reshape_tensor.shape)

    bbox_sum = tf.reduce_max(in_tensor[...,0:3], axis = -1, name  = 'bbox_sum')
    print(' bbox sum shape: ', bbox_sum.shape)

    bboxes = tf.concat([tf.round(reshape_tensor[...,0:4]), reshape_tensor[...,4:]], axis = -1, name='nz_boxes')
    print('boxes shape', bboxes.shape)


    gauss_scatt_shape = KB.int_shape(gauss_scatt)
    print(gauss_scatt_shape[-2:], type(gauss_scatt_shape))
    gauss_scatt_reshape = KB.reshape(gauss_scatt, (-1, gauss_scatt_shape[-2], gauss_scatt_shape[-1]))
    print(gauss_scatt_reshape.shape)
    # ones_map = tf.ones([384,128,128])

    return  gauss_sum, gauss_norm    # [gauss_sum, gauss_scatt, means, covar]

     
class CHMLayer(KE.Layer):
    '''
    Contextual Heatmap Layer  (previously CHMLayerTF)
    Receives the bboxes, their repsective classification and roi_outputs and 
    builds the per_class tensor

    Returns:
    -------
    The CHM layer returns the following tensors:

    pred_tensor :       [batch, NUM_CLASSES, TRAIN_ROIS_PER_IMAGE    , (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
                                in normalized coordinates   
    pred_cls_cnt:       [batch, NUM_CLASSES] 
    gt_tensor:          [batch, NUM_CLASSES, DETECTION_MAX_INSTANCES, (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
    gt_cls_cnt:         [batch, NUM_CLASSES]
    
     Note: Returned arrays might be zero padded if not enough target ROIs.
    
    '''

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        print('\n>>> CHM Layer  ')
        self.config = config

        
    def call(self, inputs):
        
        print('   > CHMLayer Call() ', len(inputs))
        # mrcnn_class , mrcnn_bbox,  output_rois, gt_class_ids, gt_bboxes, tgt_class_ids, tgt_deltas = inputs
        mrcnn_class , mrcnn_bbox,  output_rois, tgt_class_ids, gt_class_ids, gt_bboxes = inputs
        print('     mrcnn_class.shape    :',   mrcnn_class.shape, KB.int_shape(  mrcnn_class ))
        print('     mrcnn_bbox.shape     :',    mrcnn_bbox.shape, KB.int_shape(   mrcnn_bbox )) 
        print('     output_rois.shape    :',   output_rois.shape, KB.int_shape(  output_rois )) 
        print('     tgt_class_ids.shape  :', tgt_class_ids.shape, KB.int_shape(tgt_class_ids )) 
        print('     gt_class_ids.shape   :',  gt_class_ids.shape, KB.int_shape( gt_class_ids ))  
        print('     gt_bboxes.shape      :',     gt_bboxes.shape, KB.int_shape(    gt_bboxes )) 
        # print('     tgt_deltas.shape     :',    tgt_deltas.shape, KB.int_shape(   tgt_deltas )) 

        pred_tensor  = build_predictions_tf(mrcnn_class, output_rois, self.config)
        pred_cls_cnt = KL.Lambda(lambda x: tf.count_nonzero(x[:,:,:,-1],axis = -1), name = 'pred_cls_count')(pred_tensor)        
                     
        gt_tensor    = build_ground_truth_tf (gt_class_ids,  gt_bboxes, self.config.DETECTION_MAX_INSTANCES, self.config)  
        gt_cls_cnt   = KL.Lambda(lambda x: tf.count_nonzero(x[:,:,:,-1],axis = -1), name = 'gt_cls_count')(gt_tensor)
                     
        # gt_deltas    = build_ground_truth_tf(tgt_class_ids, tgt_deltas, self.config.TRAIN_ROIS_PER_IMAGE   ,self.config)  
        
        pred_heatmap_norm, pred_heatmap_scores  = build_heatmap(pred_tensor, self.config, names = ['pred_heatmap'])
        gt_heatmap_norm  , gt_heatmap_scores    = build_heatmap(gt_tensor  , self.config, names = ['gt_heatmap'])



        print('     pred_cls_cnt shape : ', pred_cls_cnt.shape       ,'Keras tensor ', KB.is_keras_tensor(pred_cls_cnt) )
        print('     gt_cls_cnt shape   : ', gt_cls_cnt.shape         ,'Keras tensor ', KB.is_keras_tensor(gt_cls_cnt) )            
        print('\n    Output build_heatmap ')
        print('     pred_heatmap_norm  : ', pred_heatmap_norm.shape  , 'Keras tensor ', KB.is_keras_tensor(pred_heatmap_norm ))
        print('     pred_heatmap_scores: ', pred_heatmap_scores.shape, 'Keras tensor ', KB.is_keras_tensor(pred_heatmap_scores))
        print('     gt_heatmap_norm    : ', gt_heatmap_norm.shape    , 'Keras tensor ', KB.is_keras_tensor(gt_heatmap_norm))
        print('     gt_heatmap_scores  : ', gt_heatmap_scores.shape  , 'Keras tensor ', KB.is_keras_tensor(gt_heatmap_scores))
        print('     complete')
        
        return [ pred_heatmap_norm , gt_heatmap_norm , pred_heatmap_scores, gt_heatmap_scores, pred_tensor, gt_tensor]

        
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        return [
                 (None, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1], self.config.NUM_CLASSES)     # pred_heatmap_norm
              ,  (None, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1], self.config.NUM_CLASSES)     # gt_heatmap_norm
              ,  (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE   , 8)                    # pred_heatmap_scores (expanded) 
              ,  (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, 8)                    # gt_heatmap+scores   (expanded) 
              ,  (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE   , 6)                    # pred_tensor
              ,  (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, 6)                    # gt_tensor   (expanded) 

              # ,  (None, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1], self.config.NUM_CLASSES)     # pred_heatmap_norm
              # ,  (None, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1], self.config.NUM_CLASSES)     # gt_heatmap_norm
              # ,  (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE   , 6)                    # gt_deltas 

              ]

        
class CHMLayerInference(KE.Layer):
    '''
    Contextual Heatmap Layer  - Inference mode (previously PCILayerTF)    
    Receives the bboxes, their repsective classification and roi_outputs and 
    builds the per_class tensor

    Returns:
    -------
    The CHM Inference layer returns the following tensors:

    pred_tensor :       [batch, NUM_CLASSES, TRAIN_ROIS_PER_IMAGE    , (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
                                in normalized coordinates   
    pred_cls_cnt:       [batch, NUM_CLASSES] 
    gt_tensor:          [batch, NUM_CLASSES, DETECTION_MAX_INSTANCES, (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
    gt_cls_cnt:         [batch, NUM_CLASSES]
    
     Note: Returned arrays might be zero padded if not enough target ROIs.
    
    '''

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        print('\n>>> CHM Layer TF ')
        self.config = config

        
    def call(self, inputs):

        print('   > PCI Layer TF: call ', type(inputs), len(inputs))
        mrcnn_class , mrcnn_bbox,  detections = inputs        
        print('     mrcnn_class.shape    :',  KB.int_shape(mrcnn_class))
        print('     mrcnn_bbox.shape     :',  KB.int_shape(mrcnn_bbox)) 
        print('     detections.shape     :',  KB.int_shape(detections)) 

        pred_tensor , _  = build_predictions_tf(mrcnn_class, mrcnn_bbox, output_rois, self.config)
        pred_heatmap  = build_heatmap_inference(detections, self.config, names = ['detections'])
            
        print('\n    Output build_heatmap ')
        print('     pred_heatmap: ', pred_heatmap.shape, 'Keras tensor ', KB.is_keras_tensor(pred_heatmap) )

        return pred_heatmap
        

        
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        return [
                (None, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1], self.config.NUM_CLASSES)     # pred_heatmap
               ]

          
          
##----------------------------------------------------------------------------------------------------------------------          
##----------------------------------------------------------------------------------------------------------------------          
##----------------------------------------------------------------------------------------------------------------------          
##----------------------------------------------------------------------------------------------------------------------          
##----------------------------------------------------------------------------------------------------------------------          
          
        # , pred_scatter  # , pred_means # , pred_covar  # , pred_tensor   # , pred_cls_cnt
        # , gt_scatter    # , gt_means   # , gt_covar    # , gt_tensor     # , gt_cls_cnt
          
          
          # , (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, \
                                            # self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1])    # pred_scatter
          # ,  (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, 2)                     # means
          # ,  (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, 2)                     # covar
          
          # ,  (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, 8)                       # pred_tensors 
          # ,  (None, self.config.NUM_CLASSES)                                                            # pred_cls_cnt

          #	,  (None, self.config.NUM_CLASSES, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1])     # gt_heatmap
          # ,  (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, \
                                            # self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1])    # gt_scatter
          # ,  (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, 2)                  # gt_means
          # ,  (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, 2)                  # gt_covar
          # ,  (None, self.config.NUM_CLASSES)                                                            # gt_cls_cnt

       #     ]
##----------------------------------------------------------------------------------------------------------------------          
## removed 16-05-2018 and replaced with more efficient method that doesnt require dynamic partitioning
##----------------------------------------------------------------------------------------------------------------------          
 
def old_build_heatmap(in_tensor, config, names = None):

    num_detections  = config.DETECTION_MAX_INSTANCES
    img_h, img_w    = config.IMAGE_SHAPE[:2]
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES  
    print('\n ')
    print('  > build_heatmap() for ', names )
    
    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES
    
    print('    orignal in_tensor shape : ', in_tensor.shape)   
    # in_tensor = in_tensor[:,:,:,2:7]
    print('    modified in_tensor shape : ', in_tensor.get_shape())
    
    rois_per_image  = tf.to_int32(in_tensor.shape[2])
    # strt_cls        = 0 if rois_per_image == 32 else 1
    print('    num of bboxes per class is : ', rois_per_image, type(rois_per_image))


    #-----------------------------------------------------------------------------
    ## Build mesh-grid to hold pixel coordinates  
    #-----------------------------------------------------------------------------
    X = tf.range(img_w, dtype=tf.int32)
    Y = tf.range(img_h, dtype=tf.int32)
    X, Y = tf.meshgrid(X, Y)
    # print('    X/Y shapes :',  X.get_shape(), Y.get_shape())
    # print('    X : \n',X.eval())
    # print('    Y : \n',Y.eval())

    # duplicate (repeat) X and Y into a  batch_size x rois_per_image tensor
    ones = tf.ones([batch_size, rois_per_image,1, 1], dtype = tf.int32)
    rep_X = ones * X
    rep_Y = ones * Y 
    # print('    Ones: ',ones.shape)                
    # print(' ones_exp * X', ones.shape, '*', X.shape, '= ',rep_X.shape)
    # print(' ones_exp * Y', ones.shape, '*', Y.shape, '= ',rep_Y.shape)

    # # stack the X and Y grids 
    bef_pos = tf.to_float(tf.stack([rep_X,rep_Y], axis = -1))
    # print(' before transpse ', bef_pos.get_shape())
    pos_grid = tf.transpose(bef_pos,[2,3,0,1,4])
    print('    after transpose ', pos_grid.get_shape())    

    # pt2_reshape = tf.reshape( in_tensor , [batch_size, num_classes * rois_per_image ,8])
    # print('    pt2_reshape shape is : ', pt2_reshape.get_shape())
    # print(pt2_reshape[0].eval())
    # print(pt2_reshape[1].eval())
    # print(pt2_reshape[2].eval())

    #-----------------------------------------------------------------------------    
    ## Stack non_zero bboxes from in_tensor into pt2_dense 
    # pt2_ind shape is [?, 3]. 
    #   pt2_ind[0] corresponds to image_index 
    #   pt2_ind[1] corresponds to class_index 
    #   pt2_ind[2] corresponds to roi row_index 
    # pt2_dense shape is [?, 6]
    #    pt2_dense[0] is image index
    #    pt2_dense[1:4]  roi cooridnaytes 
    #    pt2_dense[5]    is class id 

    #-----------------------------------------------------------------------------
    pt2_sum = tf.reduce_sum(tf.abs(in_tensor[:,:,:,:-2]), axis=-1)
    print('    pt2_sum shape ',pt2_sum.shape)
    # print(pt2_sum[0].eval())
    pt2_ind = tf.where(pt2_sum > 0)

    ## replaced the two operations with the one above
    # pt2_mask = tf.greater(pt2_sum , 0)
    # pt2_ind  = tf.where(pt2_mask)

    # print(' pt2_mask shape ', pt2_mask.get_shape())
    # print(pt2_mask.eval())
    # print('    pt2_ind shape ', pt2_ind.get_shape())
    # print(pt2_ind.eval())
    # pt2_ind_float  =  tf.to_float(pt2_ind[:,0:1])

    pt2_dense = tf.gather_nd( in_tensor, pt2_ind)

    # append image index to front of rows - REMOVED 1-5-2018
    # pt2_ind[:,0] is the same informaiton and is used in dynamic_partition    
    #  pt2_dense = tf.concat([tf.to_float(pt2_ind[:,0:1]), pt2_dense],axis=1)
    print('    dense shape ',pt2_dense.get_shape())
    # print(dense.eval())

    ## we want to slice pt2._dense by Batch size.
    ## split pt2_dense by pt2_ind[:,0], which identifies the image 
    stacked_list = tf.dynamic_partition(pt2_dense, tf.to_int32(pt2_ind[:,0]), num_partitions = batch_size )
    
    #-----------------------------------------------------------------------------
    ##  Build Stacked output from dynamically partitioned lists 
    #-----------------------------------------------------------------------------
    print('    Build Stacked output from dynamically partitioned lists --------------')  

    stacked_output=[]
    for img, item  in enumerate(stacked_list) : 
        rois_in_image  = tf.shape(item)[0]
        pad_item =  tf.pad(item,[[0, rois_per_image - rois_in_image ],[0,0]])
        stacked_output.append(pad_item)
    stacked_tensor = tf.stack(stacked_output)

    # print()    
    # print('   -- Stacked output contents --------------')    
    # print('    stacked_output shape : ', len(stacked_output))
    # for img, item  in enumerate(stacked_output) :
        # print('   img ', img, ' stacked_list[img] ', tf.shape(item).eval() ) 
    # print('   stacked_tensor shape : ', tf.shape(stacked_tensor).eval())

    #-----------------------------------------------------------------------------
    ##  Build mean and convariance tensors for Multivariate Normal Distribution 
    #-----------------------------------------------------------------------------
    width  = stacked_tensor[:,:,3] - stacked_tensor[:,:,1]      # x2 - x1
    height = stacked_tensor[:,:,2] - stacked_tensor[:,:,0]
    cx     = stacked_tensor[:,:,1] + ( width  / 2.0)
    cy     = stacked_tensor[:,:,0] + ( height / 2.0)
    means  = tf.stack((cx,cy),axis = -1)
    covar  = tf.stack((width * 0.5 , height * 0.5), axis = -1)
    covar  = tf.sqrt(covar)

    tfd = tf.contrib.distributions
    mvn = tfd.MultivariateNormalDiag( loc  = means,  scale_diag = covar)
    prob_grid = mvn.prob(pos_grid)
    prob_grid = tf.transpose(prob_grid,[2,3,0,1])

    # print('    means shape :', means.get_shape(),' covar shape ', covar.get_shape())
    # print('    from MVN    : mns shape      : ', means.shape, means.get_shape())
    # print('    from MVN    : cov shape      : ', covar.shape, covar.get_shape())
    # print('    from MVN    : mean shape     : ', mvn.mean().get_shape(), '\t stddev shape', mvn.stddev().get_shape())
    # print('    from MVN    : mvn.batch_shape: ', mvn.batch_shape , '\t mvn.event_shape ',  mvn.event_shape)
    # print('    from Linear : op shape       : ', mvn.scale.shape, ' Linear Op batch shape ',mvn.scale.batch_shape)
    # print('    from Linear : op Range Dim   : ', mvn.scale.range_dimension)
    # print('    from Linear : op Domain Dim  : ', mvn.scale.domain_dimension) 
    print('    >> input to MVN.PROB: pos_grid (meshgrid) shape: ', pos_grid.get_shape())
    print('    << output probabilities shape:' , prob_grid.get_shape())
    # print(prob_grid.eval())
    
    #--------------------------------------------------------------------------------
    # kill distributions of NaN boxes (resulting from bboxes with height/width of zero
    # which cause singular sigma cov matrices
    #--------------------------------------------------------------------------------
    gauss_grid = tf.where(tf.is_nan(prob_grid),  tf.zeros_like(prob_grid), prob_grid)

    
    ## scatter out the probability distributions based on class --------------------------
    print('\n    Scatter out the probability distributions based on class --------------')     
    class_inds      = tf.to_int32(stacked_tensor[:,:,-2])   # - should be -2 since class moved to that postion
    batch_grid, roi_grid = tf.meshgrid( tf.range(batch_size, dtype=tf.int32), tf.range(rois_per_image, dtype=tf.int32),
                                        indexing = 'ij' )
    scatter_classes = tf.stack([batch_grid, class_inds, roi_grid ],axis = -1)
    gauss_scatt     = tf.scatter_nd(scatter_classes, gauss_grid, [batch_size, num_classes, rois_per_image, img_w, img_h])

    print('    gaussian_grid      : ', gauss_grid.shape)    
    print('    class shape        : ', class_inds.shape)
    print('    roi_grid shape     : ', roi_grid.get_shape() )
    print('    batch_grid shape   : ', batch_grid.get_shape())
    print('    scatter_classes    : ', scatter_classes.get_shape())
    print('    gaussian scattered : ', gauss_scatt.shape)   
    
    ## heatmap: sum gauss_scattered based on class ---------------------------------------
    print('\n    Reduce sum based on class ---------------------------------------------')         
    gauss_sum = tf.reduce_sum(gauss_scatt, axis=2, name='pred_heatmap')
    gauss_sum = tf.where(gauss_sum > 1e-6, gauss_sum,tf.zeros_like(gauss_sum))
    gauss_sum = tf.transpose(gauss_sum,[0,2,3,1], name = names[0])
    print('    gaussian sum type/name : ', type(gauss_sum), gauss_sum.name, names[0])
    print('    gaussian_sum shape     : ', gauss_sum.get_shape(), 'Keras tensor ', KB.is_keras_tensor(gauss_sum) )    

    ## heatmap: L2 normalization  -----------------------------------------------------------------
    print('\n    L2 normalization ------------------------------------------------------')   
    heatmap_shape=KB.shape(gauss_sum)
    print(' pred_shape: KB.shape:' , heatmap_shape, ' tf.get_shape(): ', heatmap_shape.get_shape(), ' pred_maks.shape:', 
                                     gauss_sum.shape, 'tf.shape :', tf.shape(gauss_sum))
   
    gauss_flatten = KB.reshape(gauss_sum, (heatmap_shape[0], -1, heatmap_shape[-1]) )
    output_norm   = KB.l2_normalize(gauss_flatten, axis = 1)    
    gauss_norm    = KB.identity(KB.reshape(output_norm,  heatmap_shape ) , name = names[0]+'_norm')   

    print('   gauss_flatten    : ', KB.int_shape(gauss_flatten) , gauss_flatten.get_shape(),' Keras tensor ', KB.is_keras_tensor(gauss_flatten) )
    print('   gauss_norm1      : ', KB.int_shape(output_norm)   , output_norm.get_shape(),' Keras tensor ', KB.is_keras_tensor(output_norm) )
    print('   gauss_norm final : ', KB.int_shape(gauss_norm)    , gauss_norm.get_shape(),' Keras tensor ', KB.is_keras_tensor(gauss_norm) )
    ## generate score based on gaussian using bouding box masks ----------------------------------
    # flatten guassian scattered and pred_tensor, and pass on to build_bbox_score routine 
    
    # print('    complete')

    return  gauss_sum, gauss_norm    # [gauss_sum, gauss_scatt, means, covar]
    
       