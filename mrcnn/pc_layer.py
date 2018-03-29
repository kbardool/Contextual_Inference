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

def build_predictions(mrcnn_class, mrcnn_bbox, norm_output_rois, gt_class_ids, norm_gt_bboxes, config):
    # // pass model to TensorBuilder
    num_images  = config.BATCH_SIZE
    num_classes = config.NUM_CLASSES
    num_rois    = config.TRAIN_ROIS_PER_IMAGE
    num_max_gt  = config.DETECTION_MAX_INSTANCES
    h, w        = config.IMAGE_SHAPE[:2]
    num_cols    = 8 

    # mdl_outputs[outroi_idx] returns the normalized coordinates, we multiply by h,w to get true coordinates
    pred_tensor = np.zeros((num_images, num_classes, num_rois, num_cols ), dtype=np.float32)      # img_in_batch, 4, 32, 8
    pred_cls_cnt= np.zeros((num_images, num_classes), dtype=np.int16)
    output_rois = norm_output_rois * np.array([h,w,h,w])   
    
    print('mrcnn_class shape : ', mrcnn_class.shape, 'mrcnn_bbox.shape : ', mrcnn_bbox.shape)
    print('output_rois.shape : ', output_rois.shape, 'pred_tensor shape: ', pred_tensor.shape  )

    #---------------------------------------------------------------------------
    # use the argmaxof each row to determine the dominating (predicted) class
    #---------------------------------------------------------------------------
    _pred_class  =  np.argmax(mrcnn_class,axis=2)   # (32,)
    print(_pred_class)
    
    for img in range(num_images):
        for cls in range(num_classes) :
            _class_idxs = np.argwhere( _pred_class[img,:] == cls )
            pred_cls_cnt[img,cls] = _class_idxs.shape[0] 
            # print(' img/cls is: ' , img,'/',cls, '_class_idxs: ' , np.squeeze(_class_idxs))
            for j , c_idx in enumerate(_class_idxs):
                pred_tensor[img, cls, j,  0]  = j
                pred_tensor[img, cls, j,  1]  = np.max(mrcnn_class[img, c_idx ])      # probability
                pred_tensor[img, cls, j,2:6]  = output_rois[img,c_idx]                         # roi coordinates
                pred_tensor[img, cls, j,  6]  = cls                                   # class_id
                pred_tensor[img, cls, j,  7]  = c_idx                               # index from mrcnn_class array (temp for verification)               
            # sort each class in descending prediction order 
            order = pred_tensor[img,cls,:,1].argsort()
            pred_tensor[img, cls,:,1:] = pred_tensor[img, cls, order[::-1] ,1:]      #[img, cls,::-1]

    return  [pred_tensor, pred_cls_cnt]


    
def build_ground_truth(gt_class_ids, norm_gt_bboxes, config):
    # // pass model to TensorBuilder
    num_images  = config.BATCH_SIZE
    num_classes = config.NUM_CLASSES
    num_rois    = config.TRAIN_ROIS_PER_IMAGE
    num_max_gt  = config.DETECTION_MAX_INSTANCES
    h, w        = config.IMAGE_SHAPE[:2]
    num_cols    = 8 

    gt_tensor   = np.zeros((num_images, num_classes, num_max_gt, num_cols ), dtype=np.float32)      # img_in_batch, 4, 32, 8  
    gt_cls_cnt  = np.zeros((num_images, num_classes), dtype=np.int16)
    gt_bboxes   = norm_gt_bboxes   * np.array([h,w,h,w])   
    
        # gt_masks   = sample_x[gtmsk_idx][0,:,:,nz_idx]


    #---------------------------------------------------------------------------           
    #  generate ground truth tensors 
    # note - we ignore the background (class 0) in the ground truth
    #---------------------------------------------------------------------------
    for img in range(num_images):
        for cls in range(1, num_classes) :
            _class_idxs = np.argwhere( gt_class_ids[img, :] == cls)
            gt_cls_cnt[img, cls] = _class_idxs.shape[0] 
            # print('img is: ' , img , 'class: ', cls,  '_class_idxs: ' , np.squeeze(_class_idxs))
            for j , c_idx in enumerate(_class_idxs):        
                gt_tensor[img, cls, j,  0]  = j
                gt_tensor[img, cls, j,  1]  = 1.0                                 # probability
                gt_tensor[img, cls, j, 2:6] = gt_bboxes[img,c_idx,:]                         # roi coordinates
                gt_tensor[img, cls, j,  6]  = cls                                 # class_id
                gt_tensor[img, cls, j,  7]  = c_idx                               # index from mrcnn_class array (temp for verification)
            # print(gt_tensor[img,cls])

    return  [gt_tensor, gt_cls_cnt]




    
def get_stacked(in_tensor, in_cls_cnt, config):
    # print('gt_stacked: input _cs_cnt type/shape' , type(in_cls_cnt), in_cls_cnt.shape)
    _stacked = []

    for img in range(config.BATCH_SIZE):
        _substack = np.empty((0,8),dtype=np.float32)
        for cls in range(config.NUM_CLASSES):
            # if  in_cls_cnt[img, cls] > 0:
                # _substack.append(in_tensor[img, cls, 0: in_cls_cnt[img, cls]] )   
            _substack = np.vstack((_substack, in_tensor[img, cls, 0 : in_cls_cnt[img, cls]] ))   
        _stacked.append(np.asarray(_substack))                

    return _stacked
    

    
    
def build_gaussian_np(in_tensor, in_cls_cnt, config):
    from scipy.stats import  multivariate_normal
    print(' Build Gaussian NP ==========================')
    print('    in_tensor shape is ', in_tensor.shape)
    
    
    img_h, img_w = config.IMAGE_SHAPE[:2]
    num_images   = config.BATCH_SIZE
    num_classes  = config.NUM_CLASSES  
    
    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES    
    rois_per_image   = in_tensor.shape[2]     
    print(' num of bboxes per class is : ', rois_per_image)
    strt_cls = 0 if rois_per_image == 32 else 1
    
    # if rois_per_image == 100:
        # print('\n',in_tensor[0,0])
        # print('\n',in_tensor[0,1])
        # print('\n',in_tensor[0,2])
        # print('\n',in_tensor[0,3])
    # Build mesh-grid to hold pixel coordinates ----------------------------------
    X = np.arange(0, img_w, 1)
    Y = np.arange(0, img_h, 1)
    X, Y = np.meshgrid(X, Y)
    pos  = np.empty((rois_per_image,) + X.shape + (2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    # print(' pos.shape is ',pos.shape)
    pos[:,:,:,0] = X;
    pos[:,:,:,1] = Y;

    # Build the covariance matrix ------------------------------------------------
    cov = np.zeros((rois_per_image ,2))* np.array([12,19])
    # k_sess = KB.get_session()    
    
    in_stacked = get_stacked(in_tensor, in_cls_cnt, config)    
    print('   _stacked length is ', len(in_stacked), ' shape is ', in_stacked[0].shape)
    
    Zout  = np.zeros((num_images, num_classes, img_w, img_h), dtype=np.float32)
    # print(' COVARIANCE SHAPE:',cov.shape)
    # print(' Pred Tensor  is :', _tensor)
    # print('PRT SHAPES:', pred_stacked[0].shape, pred_stacked[1].shape)   
    
    for img in range(num_images):
        psx     = in_stacked[img]   #.eval(session = k_sess)  #     .eval(session=k_sess)
        # remove bboxes with zeros  
        print(' ps shape _: ',psx.shape)            
        print(' psx: ', psx)    
        ps = psx[~np.all(psx[:,2:6] == 0, axis=1)]
        print(' ps : ',ps)            
    
        width  = ps[:,5] - ps[:,3]
        height = ps[:,4] - ps[:,2]
        cx     = ps[:,3] + ( width  / 2.0)
        cy     = ps[:,2] + ( height / 2.0)
        means  = np.stack((cx,cy),axis = -1)
        cov    = np.stack((width * 0.5 , height * 0.5), axis = -1)
        # print('cov ', cov)
        #--------------------------------------------------------------------------------
        # kill boxes with height/width of zero which cause singular sigma cov matrices
        # zero_boxes = np.argwhere(width+height == 0)
        # print('zero boxes ' , zero_boxes) 
        # cov[zero_boxes] = [1,1]
        # print('cov ', cov)
        
        # print(ps.shape, type(ps),width.shape, height.shape, cx.shape, cy.shape)
        
        print('means.shape:', means.shape, 'cov.shape ', cov.shape, )
        rv  = list( map(multivariate_normal, means, cov))
        # print(' size of rv is ', len(rv))
        pdf = list( map(lambda x,y: x.pdf(y) , rv, pos))
        # print(' size of pdf is ', len(pdf))
        pdf_arr = np.asarray(pdf)       # PDF_ARR.SHAPE = # detection rois per image X  image_width X image_height
        # print('pdf_arr.shape ,: ' ,pdf_arr.shape)


        for cls in range(strt_cls, num_classes):
            _class_idxs = np.argwhere(ps[:,6] == cls) 
    #       ps = _ps[cls_idxs,:]        
            print('img: ', img,' cls:',cls,' ',np.squeeze(_class_idxs))
            # pdf_sum = np.sum(pdf_arr[_class_idxs],axis=0)
            # pdf_sumx[ ] = np.sum()
            # print('pdf_SUM.shape ,: ' ,pdf_sum.shape)
            Zout[img,cls] += np.sum(pdf_arr[_class_idxs],axis=0)[0]

    # print('Zout shape:',Zout.shape)
    # print(Zout)
    # if rois_per_image == 100:
        # print('Zout[0,0]\n',Zout[0,0])
        # print('Zout[0,1]\n',Zout[0,1])
        # print('Zout[0,2]\n',Zout[0,2])
        # print('Zout[0,3]\n',Zout[0,3])
        # print('Zout[1,0]\n',Zout[1,0])
        # print('Zout[1,1]\n',Zout[1,1])
        # print('Zout[1,2]\n',Zout[1,2])
        # print('Zout[1,3]\n',Zout[1,3])
     
    return Zout
    
    
    

class PCNLayer(KE.Layer):
    """
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
    
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        print('>>> PCN Layer : initialization')
        self.config = config

        
    def call(self, inputs):
        
        print('>>> PCN Layer : call')
        print('     mrcnn_class.shape    :',  inputs[0].shape, type(inputs[0]))
        print('     mrcnn_bbox.shape     :',  inputs[1].shape, type(inputs[1])) 
        print('     output_rois.shape    :',  inputs[2].shape, type(inputs[2])) 
    
        def wrapper(mrcnn_class, mrcnn_bbox, output_rois, gt_class_ids, gt_bboxes):
        
            pcn_tensor, pcn_cls_cnt = \
                    build_predictions(mrcnn_class, mrcnn_bbox, output_rois, self.config)
            print('pcn_tensor : ', pcn_tensor.shape)
            # print(pcn_tensor)
            print('pcn_cls_cnt: ', pcn_cls_cnt.shape)
            # print(pcn_cls_cnt)
            print(' Build Gaussian np for detected rois =========================')    
            pcn_gaussian = build_gaussian_np(pcn_tensor, pcn_cls_cnt, self.config)

            gt_tensor, gt_cls_cnt = build_ground_truth(gt_class_ids, gt_bboxes, self.config)
            
            print('gt_tensor  : ', gt_tensor.shape)
            # print( gt_tensor)
            print('gt_cls_cnt : ', gt_cls_cnt.shape)
            # print( gt_cls_cnt)
            print(' Build Gaussian np for ground_truth ==========================')    
            gt_gaussian  = build_gaussian_np(gt_tensor , gt_cls_cnt, self.config)
            
            # pcn_tensor   = tf.convert_to_tensor(pred_tensor , name = 'pcn_tensor')
            # pcn_cls_cnt  = tf.convert_to_tensor(pred_cls_cnt, name = 'pcn_cls_cnt')
            # pcn_gaussian = tf.convert_to_tensor(res         , name = 'pcn_gaussian')
            print('\n pred_tensor ', type(pcn_tensor)  , pcn_tensor.shape ) 
            print('\n pred_cls_cnt', type(pcn_cls_cnt) , pcn_cls_cnt.shape) 
            print('\n pc_gaussian ', type(pcn_gaussian), pcn_gaussian.shape)
            print('\n gt_gaussian ', type(gt_gaussian) , gt_gaussian.shape)
            print('\n gt_tensor   ', type(gt_tensor)   , gt_tensor.shape  ) 
            print('\n gt_cls_cnt  ', type(gt_cls_cnt)  , gt_cls_cnt.shape )

            # # Stack detections and cast to float32
            # # TODO: track where float64 is introduced
            # detections_batch = np.array(detections_batch).astype(np.float32)
            # # Reshape output
            # # [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
            return pcn_gaussian, gt_gaussian, pcn_tensor, pcn_cls_cnt,  gt_tensor, gt_cls_cnt

            # return np.reshape(detections_batch, [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

        # Return wrapped function
        print('>>> PCN Layer : call end  ')

        return tf.py_func(wrapper, inputs, [tf.float32, tf.float32, tf.float32, tf.int16, tf.float32, tf.int16])
        

        
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        return [
            (None, self.config.NUM_CLASSES, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1]),   
            (None, self.config.DETECTION_MAX_INSTANCES, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1]),   
            (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, 8),       # pred_tensors 
            (None, self.config.NUM_CLASSES),                                            # pred_cls_cnt
            (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, 8),       # pred_tensors 
            (None, self.config.NUM_CLASSES),                                            # pred_cls_cnt
            (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, 8),     # gt_tensors             
            (None, self.config.NUM_CLASSES)                                             # gt_cls_cnt
        ]

        
        
        
class PCILayer(KE.Layer):
    """
    Receives the bboxes, their repsective classification and roi_outputs and 
    builds the per_class tensor

    Returns:
    -------
    The PCI layer returns the following tensors:

    pred_tensor :       [batch, NUM_CLASSES, TRAIN_ROIS_PER_IMAGE    , (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
                                in normalized coordinates
    pred_gaussian       [batch, NUM_CLASSES, img_height, img_width ]
    pred_cls_cnt:       [batch, NUM_CLASSES] 
    
     Note: Returned arrays might be zero padded if not enough target ROIs.
    
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        print('>>> PCI Layer : initialization')
        self.config = config

        
    def call(self, inputs):
        
        print('>>> PCI Layer : call')
        print('     mrcnn_class.shape    :',  inputs[0].shape, type(inputs[0]))
        print('     mrcnn_bbox.shape     :',  inputs[1].shape, type(inputs[1])) 
        print('     output_rois.shape    :',  inputs[2].shape, type(inputs[2])) 
    
        def wrapper(mrcnn_class, mrcnn_bbox, output_rois):
        
            pcn_tensor, pcn_cls_cnt , gt_tensor, gt_cls_cnt = \
                    build_predictions(mrcnn_class, mrcnn_bbox, output_rois, self.config)
            print('pcn_tensor : ', pcn_tensor.shape)
            # print(pcn_tensor)
            print('pcn_cls_cnt: ', pcn_cls_cnt.shape)
            # print(pcn_cls_cnt)
            print(' Build Gaussian np for detected rois =========================')    
            pcn_gaussian = build_gaussian_np(pcn_tensor, pcn_cls_cnt, self.config)

            return pcn_gaussian, pcn_tensor, pcn_cls_cnt

            # return np.reshape(detections_batch, [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

        # Return wrapped function
        print('>>> PCN Layer : call end  ')

        return tf.py_func(wrapper, inputs, [tf.float32, tf.float32, tf.int16])
        

        
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        return [
            (None, self.config.NUM_CLASSES, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1]),     # pci_gaussian
            (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, 8),                    # pci_tensors 
            (None, self.config.NUM_CLASSES),                                                            # pci_cls_cnt
        ]


        
        
#################################################################################################################
## not in use code, can be removed later
##
#################################################################################################################        
def old_build_gaussian( pred_tensor, pred_cls_cnt , config):
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
    img_h, img_w = config.IMAGE_SHAPE[:2]
    num_images   = config.BATCH_SIZE
    num_classes  = config.NUM_CLASSES  
    
#   print(bbox.shape)
    width  = pred_tensor[:,:,:,5] - pred_tensor[:,:,:,3]
    height = pred_tensor[:,:,:,4] - pred_tensor[:,:,:,2]
    cx     = pred_tensor[:,:,:,3] + ( width  / 2.0)
    cy     = pred_tensor[:,:,:,2] + ( height / 2.0)
    means  = np.stack((cx,cy),axis = -1)
    Zout   = np.zeros((num_images, num_classes, img_w, img_h))

#   srtd_cpb_2 = np.column_stack((srtd_cpb[:, 0:2], cx,cy, width, height ))
    X    = np.arange(0, img_w, 1)
    Y    = np.arange(0, img_h, 1)
    X, Y = np.meshgrid(X, Y)
    pos  = np.empty((num_images, num_classes,) + X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,:,:,0] = X;
    pos[:,:,:,:,1] = Y;

    print(pred_cls_cnt.shape)
    for img in range(num_images):
        for cls in range(num_classes):
            _cnt = pred_cls_cnt[img,cls]
            print('class id:  ', cls      , 'class count: ',KB.shape(pred_cls_cnt))
            print('_cnt type ' ,type(_cnt), 'shape      : ',_cnt.eval())
            for box in KB.arange(_cnt, dtype='int32'):
                
                mns = means[img,cls,box]
                print('** bbox is : ' , pred_tensor[img,cls,box])
                print('    center is ({:4f},{:4f})  width is {:4f} height is {:4f} '\
                    .format(mns[0],mns[1],width[img,cls,box],height[img,cls,box]))            
                
                rv = multivariate_normal(mns,[[12,0.0] , [0.0,19]])
                Zout[img,cls,:,:] += rv.pdf(pos[img,cls])
    res = tf.convert_to_tensor(Zout)            
    
    return Zout
    