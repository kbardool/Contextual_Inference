import os
import sys
import numpy as np
import tensorflow as tf
# import keras
import keras.backend as KB
import keras.layers as KL
import keras.engine as KE
sys.path.append('..')
import mrcnn.utils as utils
import tensorflow.contrib.util as tfc
import pprint

              
              
              
##----------------------------------------------------------------------------------------------------------------------          
##
##----------------------------------------------------------------------------------------------------------------------          
     
class FCNScoringLayer(KE.Layer):
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
        print('\n>>> FCN Scoring Layer ')
        self.config = config

        
    def call(self, inputs):
        fcn_heatmap, chm_scores = inputs
        
        print('   > FCNScoreLayer Call() ', len(inputs))
        print('     fcn_heatmap.shape    :',   fcn_heatmap.shape, KB.int_shape(fcn_heatmap))
        print('      chm_scores.shape    :',    chm_scores.shape, KB.int_shape(chm_scores )) 

        fcn_scores  = self.build_fcn_scores(fcn_heatmap, chm_scores, self.config)

        print('\n    Output build_fcn_score ')
        print('     pred_heatmap_norm  : ', fcn_scores.shape  , 'Keras tensor ', KB.is_keras_tensor(fcn_scores))
        print('     complete')
        
        return [fcn_scores]

        
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        return [
                 (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE   ,16)  # pred_heatmap_scores (expanded) 
              ]
              
              
    ##----------------------------------------------------------------------------------------------------------------------          
    ##   build_fcn_scores 
    ##----------------------------------------------------------------------------------------------------------------------          
    def build_fcn_scores(self, in_heatmap, in_scores, names = None):

        num_detections  = self.config.DETECTION_MAX_INSTANCES
        img_h, img_w    = self.config.IMAGE_SHAPE[:2]
        batch_size      = self.config.BATCH_SIZE
        num_classes     = self.config.NUM_CLASSES  
        print('\n ')
        print('  > NEW build_heatmap() for ', names )
        print('    orignal in_heatmap shape : ', in_heatmap.shape)       
        # rois per image is determined by size of input tensor 
        #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
        #   ground_truth  :   config.DETECTION_MAX_INSTANCES
        rois_per_image  = KB.int_shape(in_scores)[2] 
        # strt_cls        = 0 if rois_per_image == 32 else 1
        print('    num of bboxes per class is : ', rois_per_image )

        ##--------------------------------------------------------------------------------------------
        ## generate score based on gaussian using bounding box masks 
        ## NOTE: Score is generated on NORMALIZED gaussian distributions (GAUSS_NORM)
        ##       If want to do this on NON-NORMALIZED, we need to apply it on GAUSS_SUM
        ##--------------------------------------------------------------------------------------------
        # flatten guassian scattered and input_tensor, and pass on to build_bbox_score routine 
        in_scores_shape = tf.shape(in_scores)
        in_scores_flat  = tf.reshape(in_scores, [-1, in_scores_shape[-1]])
        bboxes = tf.to_int32(tf.round(in_scores_flat[...,0:4]))
        # print('    in_scores_shape : ', in_scores_shape.eval() )
        # print('    in_scores_flat  : ', tf.shape(in_scores_flat).eval())
        # print('    boxes shape     : ', tf.shape(bboxes).eval())
        print('    Rois per image  : ', rois_per_image)

        #--------------------------------------------------------------------------------------------------------------------------
        # duplicate GAUSS_NORM <num_roi> times to pass along with bboxes to map_fn function
        #   Here we have a choice to calculate scores using the GAUSS_SUM (unnormalized) or GAUSS_NORM (normalized)
        #   after looking at the scores and ratios for each option, I decided to go with the normalized 
        #   as the numbers are large
        #---------------------------------------------------------------------------------------------------------------------------
        dup_heatmap = tf.transpose(in_heatmap, [0,3,1,2])
        print('    heatmap original shape   : ', in_heatmap.shape)
        print('    heatmap transposed shape :',  dup_heatmap.get_shape())
        dup_heatmap = tf.expand_dims(dup_heatmap, axis =2)
        # print('    heatmap expanded shape   :',  tf.shape(dup_heatmap).eval())
        dup_heatmap = tf.tile(dup_heatmap, [1,1, rois_per_image ,1,1])
        print('    heatmap tiled            : ', dup_heatmap.get_shape())
        dup_heatmap_shape   = KB.int_shape(dup_heatmap)
        dup_heatmap         = KB.reshape(dup_heatmap, (-1, dup_heatmap_shape[-2], dup_heatmap_shape[-1]))
        # print('    heatmap flattened        : ', tf.shape(dup_heatmap).eval())

        scores = tf.map_fn(self.build_mask_routine, [dup_heatmap, bboxes], dtype=tf.float32)    

        ##--------------------------------------------------------------------------------------------
        ## Add returned values from scoring to the end of the input score 
        ##--------------------------------------------------------------------------------------------    
        # consider the two new columns for reshaping the gaussian_bbox_scores
        new_shape   = in_scores_shape + [0,0,0, tf.shape(scores)[-1]]        
        bbox_scores = tf.concat([in_scores_flat, scores], axis = -1)
        bbox_scores = tf.reshape(bbox_scores, new_shape)
        # print('    new shape is            : ', new_shape.eval())
        # print('    in_scores_flat          : ', tf.shape(in_scores_flat).eval())
        # print('    Scores shape            : ', tf.shape(scores).eval())   # [(num_batches x num_class x num_rois ), 3]
        # print('    boxes_scores (rehspaed) : ', tf.shape(bbox_scores).eval())    

        ##--------------------------------------------------------------------------------------------
        ## Normalize computed score above, and add it to the heatmap_score tensor as last column
        ##--------------------------------------------------------------------------------------------
        scr_L2norm   = tf.nn.l2_normalize(bbox_scores[...,-1], axis = -1)   # shape (num_imgs, num_class, num_rois)
        scr_L2norm   = tf.expand_dims(scr_L2norm, axis = -1)
       
        ##--------------------------------------------------------------------------------------------
        # shape of tf.reduce_max(bbox_scores[...,-1], axis = -1, keepdims=True) is (num_imgs, num_class, 1)
        #  This is a regular normalization that moves everything between [0, 1]. This causes negative values to move
        #  to -inf. 
        # To address this a normalization between [-1 and +1] was introduced. Not sure how this will work with 
        # training tho.
        ##--------------------------------------------------------------------------------------------
        scr_norm     = bbox_scores[...,-1]/ tf.reduce_max(bbox_scores[...,-1], axis = -1, keepdims=True)
        scr_norm     = tf.where(tf.is_nan(scr_norm),  tf.zeros_like(scr_norm), scr_norm)     
        
        #--------------------------------------------------------------------------------------------
        # this normalization moves values to [-1 +1] 
        #--------------------------------------------------------------------------------------------    
        # reduce_max = tf.reduce_max(bbox_scores[...,-1], axis = -1, keepdims=True)
        # reduce_min = tf.reduce_min(bbox_scores[...,-1], axis = -1, keepdims=True)  ## epsilon    = tf.ones_like(reduce_max) * 1e-7
        # scr_norm  = (2* (bbox_scores[...,-1] - reduce_min) / (reduce_max - reduce_min)) - 1     

        # scr_norm     = tf.where(tf.is_nan(scr_norm),  tf.zeros_like(scr_norm), scr_norm)  
        scr_norm     = tf.expand_dims(scr_norm, axis = -1)                             # shape (num_imgs, num_class, 32, 1)
        fcn_scores   = KB.identity(tf.concat([bbox_scores, scr_norm, scr_L2norm], axis = -1), name = 'fcn_heatmap_scores') 

        print('    fcn_scores  final shape : ', fcn_scores.shape ,' Keras tensor ', KB.is_keras_tensor(fcn_scores) )  
        print('    complete')

        return fcn_scores     
    
        
                  
    ##----------------------------------------------------------------------------------------------------------------------          
    ##
    ##----------------------------------------------------------------------------------------------------------------------          
        
    def build_mask_routine(self, input_list):
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
            mask_sum    =  tf.reduce_sum(mask)
            mask_applied = tf.multiply(heatmap_tensor, mask, name = 'mask_applied')
            bbox_area    = tf.to_float((input_row[2]-input_row[0]) * (input_row[3]-input_row[1]))
            gaussian_sum = tf.reduce_sum(mask_applied)
            ratio        = gaussian_sum / bbox_area 
            ratio        = tf.where(tf.is_nan(ratio),  0.0, ratio)  
        return tf.stack([gaussian_sum, bbox_area, ratio], axis = -1)
        
        
              