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

     
    
def build_predictions_tf(mrcnn_class, mrcnn_bbox, norm_output_rois, config):
    # // pass model to TensorBuilder
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES
    num_rois        = config.TRAIN_ROIS_PER_IMAGE
    h, w            = config.IMAGE_SHAPE[:2]
    num_cols        = 8
    print('\n    *** build_predictions_tf' )
    sess = tf.InteractiveSession()
    
    output_rois = norm_output_rois * np.array([h,w,h,w])   

    # print('>>> build_predictions_tf')
    # print('    mrcnn_class shape : ', mrcnn_class.shape)
    # print('    mrcnn_bbox.shape  : ', mrcnn_bbox.shape )
    # print('    output_rois.shape : ', output_rois.shape)
    # print('    pred_tensor       : ', pred_tensor.shape)
    # print('    pred_cls_cnt      : ', pred_cls_cnt.shape)
    #---------------------------------------------------------------------------
    # use the argmaxof each row to determine the dominating (predicted) class
    #---------------------------------------------------------------------------
    # np.set_printoptions(linewidth=100, precision=4)
    bbox_selected    = tf.zeros_like(norm_output_rois)
    pred_classes     = tf.to_int32(tf.argmax( mrcnn_class,axis=-1))
    pred_classes_exp = tf.to_float(tf.expand_dims(pred_classes ,axis=-1))
    pred_scores      = tf.reduce_max(mrcnn_class ,axis=-1, keepdims=True)   # (32,)
    # print('    pred_classes with highest scores:', pred_classes.get_shape() )
    # pred_scores_exp = tf.to_float(tf.expand_dims(pred_scores, axis=-1))
    # print('    pred_ scores:', pred_scores.get_shape())
    

    batch_grid, roi_grid = tf.meshgrid( tf.range(batch_size, dtype=tf.int32),
                                        tf.range(num_rois, dtype=tf.int32), indexing = 'ij' )
    bbox_idx             = tf.to_float(tf.expand_dims(roi_grid , axis = -1))    

    #-----------------------------------------------------------------------------------
    # This part is used if we want to gather bbox coordinates from mrcnn_bbox 
    #  Currently we are gathering bbox coordinates form output_roi so we dont need this
    #-----------------------------------------------------------------------------------
    # gather_boxes    = tf.stack([batch_grid, roi_grid, pred_classes, ], axis = -1)
    # print('-- gather_boxes  ----')
    # print('gather_boxes inds', type(gather_boxes), 'shape',tf.shape(gather_boxes).eval())
    # print(gather_boxes.eval())
    # bbox_selected   = tf.gather_nd(mrcnn_bbox, gather_boxes)
    # print('    bbox_selected shape : ', bbox_selected.get_shape())
    # print(bbox_selected[0].eval())    
    

    pred_array  = tf.concat([bbox_idx, pred_scores , output_rois, pred_classes_exp], axis=2)
    # print('    -- pred_tensor tf ------------------------------') 
    print('    pred_array shape:', pred_array.shape)
    
    # pred_array = pred_array[~np.all(pred_array[:,:,2:6] == 0, axis=1)]
    
    # class_ids = tf.to_int32(pred_array[:,:,6])
    # print('    class shape: ', class_ids.get_shape())
    # print(class_ids.eval())

    # print('    roi_grid ', type(roi_grid), 'shape', roi_grid.get_shape())
    # print(roi_grid.eval())
    # print('    batch_grid     ', type(batch_grid), 'shape',(batch_grid.get_shape()))
    # print(batch_grid.eval())

    scatter_ind          = tf.stack([batch_grid , pred_classes, roi_grid],axis = -1)
    # print('scatter_ind', type(scatter_ind), 'shape',tf.shape(scatter_ind).eval())
    pred_scatt = tf.scatter_nd(scatter_ind, pred_array, [batch_size, num_classes, num_rois,7])
    print('    pred_scatter shape is ', pred_scatt.get_shape(), pred_scatt)
    
    ## sort in each class dimension based on prediction score 

    _, sort_inds = tf.nn.top_k(pred_scatt[:,:,:,1], k=pred_scatt.shape[2])
    print('    sort inds shape : ', sort_inds.get_shape())

    # build gathering indexes to use in sorting 
    class_grid, batch_grid, roi_grid = tf.meshgrid(tf.range(num_classes),tf.range(batch_size), tf.range(num_rois))
    roi_grid_exp = tf.to_float(tf.expand_dims(roi_grid, axis = -1))
    print('    class_grid  ', type(class_grid) , 'shape', class_grid.get_shape())
    print('    batch_grid  ', type(batch_grid) , 'shape', batch_grid.get_shape())
    print('    roi_grid shape', roi_grid.get_shape(), 'roi_grid_exp shape ', roi_grid_exp.get_shape())
    
    
    gather_inds  = tf.stack([batch_grid , class_grid, sort_inds],axis = -1)
    pred_tensor  = tf.gather_nd(pred_scatt, gather_inds)
    print('    gather_inds ', type(gather_inds), 'shape', gather_inds.get_shape())
    print('    pred_tensor (gathered)  : ', pred_tensor.get_shape())

    
    print('    -- pred_tensor results (bboxes sorted by score) ----')
    pred_tensor  = tf.concat([pred_tensor, roi_grid_exp], axis = -1)
    pred_cls_cnt = tf.count_nonzero(pred_tensor[:,:,:,0], axis = -1)
    
    print('    final pred_tensor shape  : ', pred_tensor.get_shape())
    print('    final pred_cls_cnt shape : ',pred_cls_cnt.get_shape())
    print('    complete')

    return  [pred_tensor, pred_cls_cnt] 
    
    
    
def build_ground_truth_tf(gt_class_ids, norm_gt_bboxes, config):
    # // pass model to TensorBuilder
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES
    num_detections  = config.DETECTION_MAX_INSTANCES
    h, w            = config.IMAGE_SHAPE[:2]
    num_cols        = 7 
    print('\n    *** build_ground_truth_tf' )
    print('    gt_class_ids shape : ', gt_class_ids.shape, '    notm_gt_bbox.shape  : ', norm_gt_bboxes.shape )

    # sess = tf.InteractiveSession()
    gt_bboxes       = norm_gt_bboxes * np.array([h,w,h,w])   

    #---------------------------------------------------------------------------
    # use the argmaxof each row to determine the dominating (predicted) class
    #---------------------------------------------------------------------------
        
    # gt_classes     = gt_class_ids    # batch_size x max gt detections
    gt_classes_exp = tf.to_float(tf.expand_dims(gt_class_ids ,axis=-1))
    print('    gt_classes_exp shape ', gt_classes_exp.get_shape() )

    ones = tf.ones_like(gt_class_ids)
    zeros= tf.zeros_like(gt_class_ids)
    mask = tf.greater(gt_class_ids , 0)

    gt_scores  =  tf.where(mask, ones, zeros)
    # pred_scores      = tf.reduce_max(mrcnn_class ,axis=-1, keep_dims=True)   # (32,)
    gt_scores_exp = tf.to_float(tf.expand_dims(gt_scores, axis=-1))
    print('    pred_ scores shape ', gt_scores.get_shape())
    

    batch_grid, bbox_grid = tf.meshgrid( tf.range(batch_size    , dtype=tf.int32), 
                                         tf.range(num_detections, dtype=tf.int32), indexing = 'ij' )
    
    
    bbox_idx_zeros  = tf.zeros_like(bbox_grid)
    bbox_idx        = tf.where(mask, bbox_grid , bbox_idx_zeros)
    bbox_idx        = tf.to_float(tf.expand_dims(bbox_idx, axis = -1))    
    
    gt_array        = tf.concat([bbox_idx, gt_scores_exp , gt_bboxes, gt_classes_exp], axis=2)
    
    print('    bbox_idx shape    ', bbox_idx.get_shape())
    # print(bbox_idx.eval()) 
    print('    gt_array shape    ', gt_array.get_shape())
    print('    bbox_grid  shape  ', bbox_grid.get_shape())
    # print(bbox_grid.eval())
    print('    batch_grid shape  ', batch_grid.get_shape())
    # print(batch_grid.eval())


    scatter_ind = tf.stack([batch_grid , gt_class_ids, bbox_grid],axis = -1)
    gt_scatter = tf.scatter_nd(scatter_ind, gt_array, [batch_size, num_classes, num_detections,7])
    
    
    # print('-- stack results ----')
    print('    scatter_ind shape ', scatter_ind.get_shape())
    # print(scatter_ind.eval())
    print('    gt_scatter shape ', gt_scatter.get_shape())
    
    ## sort in each class dimension based on y2 (column 2)

    _ , sort_inds = tf.nn.top_k(gt_scatter[:,:,:,2], k=gt_scatter.shape[2])

    # build gathering indexes to use in sorting 
    class_grid, batch_grid, bbox_grid = tf.meshgrid(tf.range(num_classes),tf.range(batch_size), tf.range(num_detections))
    bbox_grid_exp = tf.to_float(tf.expand_dims(bbox_grid, axis = -1))

    print('    build gathering indexes to use in sorting -------')    
    print('    sort inds shape : ', sort_inds.get_shape())
    print('    class_grid  shape ', class_grid.get_shape())
    # print(class_grid.eval())
    print('    batch_grid  shape ', batch_grid.get_shape())
    # print(class_grid.eval())
    print('    bbox_grid   shape ', bbox_grid.get_shape() , ' bbox_grid_exp shape ', bbox_grid_exp.get_shape())
    # print(bbox_grid.eval())

    gather_inds = tf.stack([batch_grid , class_grid, sort_inds],axis = -1)
    gt_tensor   = tf.gather_nd(gt_scatter, gather_inds)
    gt_tensor   = tf.concat([gt_tensor, bbox_grid_exp], axis = -1)
    print('    gather_inds shape ', gather_inds.get_shape())
    print('    gt_tensor (gathered)   : ', gt_tensor.get_shape())
    
    gt_cls_cnt = tf.count_nonzero(gt_tensor[:,:,:,0],axis = -1)
    print('    final gt_tensor shape  : ', gt_tensor.get_shape())
    print('    final gt_cls_cnt shape : ', gt_cls_cnt.get_shape())
    print('    complete')

    return  [gt_tensor, gt_cls_cnt] 

    
    
    
    
def build_gaussian_tf(in_tensor, pred_cls_cnt, config):
        
    # rois_per_image  = 32
    
    num_detections  = config.DETECTION_MAX_INSTANCES
    h, w            = config.IMAGE_SHAPE[:2]
    num_cols        = 8
    img_h, img_w    = config.IMAGE_SHAPE[:2]
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES  
    print('\n    *** build_gaussian_tf ')
    ## rois per image is determined by size of input tensor 
    ##   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    ##   ground_truth  :   config.DETECTION_MAX_INSTANCES
    print('    in_tensor shape : ', in_tensor.shape)   
    in_tensor = in_tensor[:,:,:,2:7]
    print('    modified in_tensor shape : ', in_tensor.get_shape())
    
    rois_per_image   = tf.to_int32(in_tensor.shape[2])
    strt_cls = 0 if rois_per_image == 32 else 1
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

    # print('    means shape ', means.get_shape())
    # print('    covar shape ', covar.get_shape())
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
    print('    Scatter out the probability distributions based on class --------------')     
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
    print('    Reduce sum based on class ---------------------------------------------')         
    gauss_sum = tf.reduce_sum(gauss_scatt, axis=2)
    gauss_sum = tf.where(gauss_sum > 1e-6, gauss_sum,tf.zeros_like(gauss_sum))
    gauss_sum = tf.transpose(gauss_sum,[0,2,3,1])
    print('    gaussian_sum shape : ', gauss_sum.get_shape())    
    print('    complete')
    
    return  gauss_sum    # [gauss_sum, gauss_scatt, means, covar]
    

        
class PCNLayerTF(KE.Layer):
    '''
    Receives the bboxes, their repsective classification and roi_outputs and 
    builds the per_class tensor

    Returns:
    -------
    The PCN layer returns the following tensors:

    pred_tensor :       [batch, NUM_CLASSES, TRAIN_ROIS_PER_IMAGE    , (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
                                in normalized coordinates   
    pred_cls_cnt:       [batch, NUM_CLASSES] 
    gt_tensor:          [batch, NUM_CLASSES, DETECTION_MAX_INSTANCES, (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
    gt_cls_cnt:         [batch, NUM_CLASSES]
    
     Note: Returned arrays might be zero padded if not enough target ROIs.
    
    '''

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        print('\n>>> PCN Layer TF ')
        self.config = config

        
    def call(self, inputs):
        
        print('>>> PCN Layer TF: call ', type(inputs), len(inputs))
        print('     mrcnn_class.shape    :',  inputs[0].shape, type(inputs[0]))
        print('     mrcnn_bbox.shape     :',  inputs[1].shape, type(inputs[1])) 
        print('     output_rois.shape    :',  inputs[2].shape, type(inputs[2])) 
        print('     gt_class_ids.shape   :',  inputs[3].shape, type(inputs[3])) 
        print('     gt_bboxes.shape      :',  inputs[4].shape, type(inputs[4])) 
        mrcnn_class , mrcnn_bbox,  output_rois, gt_class_ids, gt_bboxes = inputs

        pred_tensor , pred_cls_cnt  = build_predictions_tf(mrcnn_class, mrcnn_bbox, output_rois, self.config)
        # print('     pred_tensor : ', pred_tensor.shape, '  pred_cls_cnt: ', pred_cls_cnt.shape)

        gt_tensor   , gt_cls_cnt    = build_ground_truth_tf(gt_class_ids, gt_bboxes, self.config)  
        # print('     gt_tensor  : ', gt_tensor.shape   , '  gt_cls_cnt  : ', gt_cls_cnt.shape)

        # print(' Build Gaussian np for detected rois =========================')    
        # pred_scatter, pred_gaussian, pred_means, pred_covar  = build_gaussian_tf(pred_tensor, pred_cls_cnt, self.config)
        pred_gaussian   = build_gaussian_tf(pred_tensor, pred_cls_cnt, self.config)
        # print('   Output build_gaussian_tf (predicitons)')
        # print('     pred_gaussian : ', pred_gaussian.shape)
        # print('     pred_scatter  : ', pred_scatter.shape)
        # print('     means         : ', pred_means.shape  , '    pred_covar    : ', pred_covar.shape)
        
        # print(' Build Gaussian np for ground_truth ==========================')    
        # gt_gaussian, gt_scatter, gt_means, gt_covar  = build_gaussian_tf(gt_tensor , gt_cls_cnt, self.config)
        gt_gaussian     = build_gaussian_tf(gt_tensor , gt_cls_cnt, self.config)
        print('   Output build_gaussian_tf (ground truth)')
        print('     gt_gaussian : ', gt_gaussian.shape)
        # print('     gt_scatter  : ', gt_scatter.shape)
        # print('     gt_means    : ', gt_means.shape  , '    gt_covar    : ', gt_covar.shape)
        
        return [ 
                  pred_gaussian
                # , pred_scatter
                # , pred_means 
                # , pred_covar
                , pred_tensor  
                , pred_cls_cnt

                , gt_gaussian  
                # , gt_scatter   
                # , gt_means 
                # , gt_covar
                , gt_tensor    
                , gt_cls_cnt
                ]


        
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        return [
            (None, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1], self.config.NUM_CLASSES)     # pred_gaussian
          # , (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, \
                                            # self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1])    # pred_scatter
          # ,  (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, 2)                     # means
          # ,  (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, 2)                     # covar
          
          ,  (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, 8)                       # pred_tensors 
          ,  (None, self.config.NUM_CLASSES)                                                            # pred_cls_cnt

          ,  (None, self.config.NUM_CLASSES, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1])     # gt_gaussian
          # ,  (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, \
                                            # self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1])    # gt_scatter
          # ,  (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, 2)                  # gt_means
          # ,  (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, 2)                  # gt_covar
          ,  (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, 8)                    # gt_tensor 
          ,  (None, self.config.NUM_CLASSES)                                                            # gt_cls_cnt

            ]
 
       