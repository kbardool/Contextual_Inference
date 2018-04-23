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

     
def build_predictions(mrcnn_class, mrcnn_bbox, norm_output_rois, config):
    # // pass model to TensorBuilder
    num_images  = config.BATCH_SIZE
    num_classes = config.NUM_CLASSES
    num_rois    = config.TRAIN_ROIS_PER_IMAGE
    h, w        = config.IMAGE_SHAPE[:2]
    num_cols    = 8 

    # print('    build_predictions (NP) ' )
    # mdl_outputs[outroi_idx] returns the normalized coordinates, we multiply by h,w to get true coordinates
    pred_tensor = np.zeros((num_images, num_classes, num_rois, num_cols ), dtype=np.float32)      # img_in_batch, 4, 32, 8
    pred_cls_cnt= np.zeros((num_images, num_classes), dtype=np.int16)
    output_rois = norm_output_rois * np.array([h,w,h,w])   
    pred_new    = np.empty((num_rois, num_cols))
    # print('mrcnn_class shape : ', mrcnn_class.shape, '\t mrcnn_bbox.shape : ', mrcnn_bbox.shape )
    # print('output_rois.shape : ', output_rois.shape, '\t pred_tensor shape: ', pred_tensor.shape  )

    #---------------------------------------------------------------------------
    # use the argmaxof each row to determine the dominating (predicted) class
    #---------------------------------------------------------------------------
    for img in range(num_images):
        img_roi_scores  = mrcnn_class[img]              # 2x32x4 -> 32x4
        img_roi_boxes   = output_rois[img] 
        # img_roi_boxes   = output_rois[~np.all(output_rois[img] != 0, axis=0)]
        # print(' before \n', output_rois[img])
        # print(' after  \n', img_roi_boxes)
        predicted_class     = np.argmax(img_roi_scores,axis=1)   # (32,)

        # print('----------------------------------------------------------')
        # print(' image: ' , img)
        # print('----------------------------------------------------------')       
        # # print('mrcnn_class[img] ',img_roi_scores.shape)
        # # print(img_roi_scores)
        # print('output_rois[img] ',img_roi_boxes.shape)
        # print(img_roi_boxes)
        # # print('img: ',img , 'pred_cls: ', _pred_class)
        
        for cls in range(num_classes) :
            cls_idxs = np.where( predicted_class == cls )
            cls_cnt  = cls_idxs[0].shape[0]
            pred_new.fill(0)
            # print('----------------------------------------------------------')
            # print(' img/cls is: ' , img,'/',cls, 'cls_idxs: ' , cls_idxs[0])
            # print(' cls_idxs[0].shape: ', cls_idxs[0].shape, ' cls_cnt',cls_cnt)
            # print('----------------------------------------------------------')
            score = np.max(img_roi_scores[cls_idxs],axis = -1)
            pred_new[:cls_cnt,0]  = cls_idxs[0]
            pred_new[:cls_cnt,1]  = score
            pred_new[:cls_cnt,2:6]= img_roi_boxes[cls_idxs]
            pred_new[:cls_cnt,6]  = cls
            pred_new[:cls_cnt,7]  = range(cls_cnt)
            

            # print(' mrcnn_class: ', img_roi_scores.shape)
            # print(  img_roi_scores[cls_idxs])
            # print(' score.shape  : ', score.shape)
            # print(  score)
            # print(' img_roi_boxes.shape',  img_roi_boxes[cls_idxs].shape)
            
            ## sort pred_new array in descending prediction order 
            order                  = pred_new[:cls_cnt,1].argsort()
            pred_new[:cls_cnt,:7]  = pred_new[order[::-1] ,:7]      #[img, cls,::-1]           
            # print('pred_new[img,cls] after sort:')
            # print(pred_new)

            
            ##  drop (0,0,0,0) bounding boxes from pred_new array just constructed 
            # class_bboxes  = pred_new[:cls_cnt,2:6]
            # vld_indices   = ~np.all( pred_new[:,2:6] == 0, axis=1)
            # non_zero_rois = np.count_nonzero(vld_indices)
            # print('vld_indices  \n' , vld_indices.shape, 'non zero bounding boxes: ', non_zero_rois)
            # print(vld_indices)

            # pred_tensor[img,cls]  = np.pad(pred_new[vld_indices], ((0, num_rois - non_zero_rois),(0,0)),'constant', constant_values = 0)
            # print('pred_new after suppression of zero bboxes  \n' , pred_tensor[img, cls])
            # pred_cls_cnt[img, cls] = non_zero_rois

            pred_tensor[img,cls]   = pred_new
            pred_cls_cnt[img, cls] = cls_cnt
            
            # print(' pred_cls_cnt is ' , pred_cls_cnt)
    return  [pred_tensor, pred_cls_cnt]


       
def build_ground_truth(gt_class_ids, norm_gt_bboxes, config):
    # // pass model to TensorBuilder
    num_images      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES
    num_detections  = config.DETECTION_MAX_INSTANCES
    h, w            = config.IMAGE_SHAPE[:2]
    num_cols        = 8 
    # print('    build_grand_truth (NP) ' )

    gt_tensor   = np.zeros((num_images, num_classes, num_detections, num_cols ), dtype=np.float32)      # img_in_batch, 4, 32, 8  
    gt_cls_cnt  = np.zeros((num_images, num_classes), dtype=np.int16)
    gt_bboxes   = norm_gt_bboxes   * np.array([h,w,h,w])   
    gt_new      = np.empty((num_detections, num_cols))
        # gt_masks   = sample_x[gtmsk_idx][0,:,:,nz_idx]
    # print('gt_class_ids shape : ', gt_class_ids.shape, '\t norm_gt_bboxes.shape : ', norm_gt_bboxes.shape )
    # print('\n',gt_class_ids)
    # print('\n',gt_bboxes)

    #---------------------------------------------------------------------------           
    #  generate ground truth tensors 
    # note - we ignore the background (class 0) in the ground truth
    #---------------------------------------------------------------------------
    for img in range(num_images):
    
        for cls in range(1, num_classes) :
     
            cls_idxs = np.where( gt_class_ids[img, :] == cls)
            cls_cnt  = cls_idxs[0].shape[0] 
            # print('img is: ' , img , 'class: ', cls,  'cls_idxs: ' , cls_idxs)
            gt_new.fill(0)
            gt_new[:cls_cnt,0]  = range(cls_cnt)
            gt_new[:cls_cnt,1]  = 1.0
            gt_new[:cls_cnt,2:6]= gt_bboxes[img, cls_idxs,:]
            gt_new[:cls_cnt,6]  = cls
            gt_new[:cls_cnt,7]  = cls_idxs[0]
            
            ## sort pred_new array in descending prediction order 
            order                  = gt_new[:cls_cnt,2].argsort()
            gt_new[:cls_cnt,:7]    = gt_new[order[::-1] ,:7]      #[img, cls,::-1]           
            # print('pred_new[img,cls] after sort:')
            # print(pred_new)
            
            gt_tensor[img,cls]   = gt_new
            gt_cls_cnt[img, cls] = cls_cnt
            # print('gt_tensor is')
    return  [gt_tensor, gt_cls_cnt]

    
    
def get_stacked(in_tensor, in_cls_cnt, config):
    # print('gt_stacked: input _cs_cnt type/shape' , type(in_cls_cnt), in_cls_cnt.shape)
    _stacked = []
    for img in range(config.BATCH_SIZE):
        _substack = np.empty((0,8),dtype=np.float32)
        for cls in range(config.NUM_CLASSES):
             _substack = np.vstack((_substack, in_tensor[img, cls, 0 : in_cls_cnt[img, cls]] ))   
        _stacked.append(np.asarray(_substack))                

    return _stacked
    

    
    
def build_gaussian_np(in_tensor, in_cls_cnt, config):
    from scipy.stats import  multivariate_normal
    img_h, img_w = config.IMAGE_SHAPE[:2]
    num_images   = config.BATCH_SIZE
    num_classes  = config.NUM_CLASSES  
    means_list   = []
    covar_list   = [] 
    gauss_sum         = np.zeros((num_images, num_classes, img_w, img_h), dtype=np.float32)
    cls_mask     = np.empty((img_w, img_h), dtype=np.int8)
    in_stacked   = get_stacked(in_tensor, in_cls_cnt, config)    
    # print(' stacked length is ', len(in_stacked), ' shape is ', in_stacked[0].shape)
    
    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES    
    rois_per_image   = in_tensor.shape[2]     
    # strt_cls = 0 if rois_per_image == 32 else 1
    strt_cls = 0

    print('   input_tensor shape is ', in_tensor.shape)
    print('   num of bboxes per class is : ', rois_per_image)

    
    # Build mesh-grid to hold pixel coordinates ----------------------------------
    X = np.arange(0, img_w, 1)
    Y = np.arange(0, img_h, 1)
    X, Y = np.meshgrid(X, Y)
    pos  = np.empty((rois_per_image,) + X.shape + (2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,:,0] = X;
    pos[:,:,:,1] = Y;

    pdf_batch_list = []
    
    for img in range(num_images):
        # print('   ===> Img: ', img)
        
        ## remove zero bounding boxes        
        psx     = in_stacked[img]   #.eval(session = k_sess)  #     .eval(session=k_sess)
        stacked_tensor = psx[~np.all(psx[:, 2:6] == 0, axis=1)]
        # remove bboxes with zeros  
        # print('  input tensor shape _: ',psx.shape)            
        # print(psx)    
        # print('  input tensor after zeros removals shape _: ',stacked_tensor.shape)            
        # print(stacked_tensor)            

        ## compute mean and covariance for gaussian distributions
        width  = stacked_tensor[:,5] - stacked_tensor[:,3]
        height = stacked_tensor[:,4] - stacked_tensor[:,2]
        cx     = stacked_tensor[:,3] + ( width  / 2.0)
        cy     = stacked_tensor[:,2] + ( height / 2.0)
        means  = np.stack((cx,cy),axis = -1)
        covar  = np.stack((width * 0.5 , height * 0.5), axis = -1)
        means_list.append(np.pad(means, ((0, rois_per_image - means.shape[0]), (0,0)), 'constant', constant_values = 0))
        covar_list.append(np.pad(covar, ((0, rois_per_image - covar.shape[0]), (0,0)), 'constant', constant_values = 0))
    
        # weight = np.ones((stacked_tensor.shape[0]))
        # print('covar ', covar)
        #--------------------------------------------------------------------------------
        # kill boxes with height/width of zero which cause singular sigma covar matrices
        # zero_boxes = np.argwhere(width+height == 0)
        # print('zero boxes ' , zero_boxes) 
        # covar[zero_boxes] = [1,1]
        # print('covar ', covar)
        # print(stacked_tensor.shape, type(stacked_tensor),width.shape, height.shape, cx.shape, cy.shape)
        
        ## define gaussian distributions and compute PDF
        # print('  img : ', img, ' means.shape:', means.shape, 'covar.shape ', covar.shape)
        rv  = list( map(multivariate_normal, means, covar))
        # print('  size of rv is ', len(rv))
        pdf = list( map(lambda x,y: x.pdf(y) , rv, pos))
        # print('  size of pdf is ', len(pdf))
        pdf_arr = np.dstack(pdf)          # PDF_ARR.SHAPE = # detection rois per image X  image_width X image_height
        # print('    pdf_arr.shape : ' ,pdf_arr.shape)
        pdf_class_list = [] 
        
        
        ##  Create a mask containing 1 for bounding box locations and 0 for all othe locations 
        for cls in range(strt_cls, num_classes):
            class_indices = np.argwhere(stacked_tensor[:,6] == cls).squeeze(axis=-1)
            # print('       img: ', img,' cls:',cls,' ',np.squeeze(class_indices))
            
            ## if no bboxes apply to this class, push an empty tensor into pdf_class_list
            if class_indices.shape[0] == 0:
                pdf_class = np.zeros((rois_per_image , img_w, img_h),dtype = np.float32)
                # print('   cls :', cls,'  pdf_class.shape : ' ,pdf_class.shape)         
                pdf_class_list.append(pdf_class) 
                continue
            
            cls_mask.fill(0)            
            class_array = stacked_tensor[class_indices,:]
            # print('    class_array shpe:',class_array.shape)            
            # print('    class_array :  \n', class_array)   
            
            
            ## build class specific mask based on bounding boxes
            class_bboxes  = class_array[:,2:6]
            valid_bboxes  = class_bboxes[~np.all(class_bboxes == 0, axis=1)]
            valid_bboxes  = np.round(valid_bboxes).astype(np.int)

            # print('valid_bboxes  \n' , valid_bboxes)
            
            ## for each bounding box set locations in the mask to 1 
            for i in valid_bboxes:
                # _tmp[ slice(i[0],i[2]) , slice(i[1],i[3]) ]
                # print('(slice(',i[0],',',i[2],'), slice(',i[1],',',i[3],'))')
                # cls_mask[ slice(i[0],i[2]) , slice(i[1],i[3]) ] += 1
                cls_mask[ slice(i[0],i[2]) , slice(i[1],i[3]) ] = 1
                
            # slice_list = [(slice(i[0],i[2]), slice(i[1],i[3])) for i in valid_bboxes]
            # print('Slice list \n')
            # pp = pprint.PrettyPrinter(indent=2, width=100)
            # pp.pprint(slice_list)
            # cls_mask = np.fill(0)
            # for i in slice_list:
                # cls_mask += 

            ## --------------------------------------------------
            pdf_arr_abs = pdf_arr[...,class_indices]
            pdf_sum_abs = np.sum(pdf_arr_abs,axis=-1)

            # generate weighted sum -- currently not used
            # nones  = np.ones_like(weight)  #  <-- currently not used 
            # norm   = np.sum(class_array[:,1])
            # weight = class_array[:,1]/norm
            # pdf_arr_wtd = pdf_arr[...,class_indices] * weight
            # pdf_sum_wtd = np.sum(pdf_arr_wtd,axis=-1)
            
            pdf_class   = np.transpose(pdf_arr_abs, axes = [2,0,1])           
            pdf_class   = np.pad(pdf_class, ((0,rois_per_image-pdf_class.shape[0]),(0,0),(0,0)), 'constant',constant_values=0)
            pdf_class_list.append(pdf_class) 
            # print('    pdf_arr_wtd shape:       ', pdf_arr_wtd.shape)
            # print('    cls :', cls,'  pdf_class.shape : ' ,pdf_class.shape)                     
            # print('    Weighted max/min ',np.max(np.max(pdf_arr_wtd),0),np.min(np.min(pdf_arr_wtd),0))
            # print('    Absolute max/min ',np.max(np.max(pdf_arr_abs),0),np.min(np.min(pdf_arr_abs),0))
            # print('    pdf_sum_wtd.shape ,: ' ,pdf_sum_wtd.shape , '   pdf_sum_abs.shape: ',pdf_sum_abs.shape)
            # print mask ---------------
            # if rois_per_image == 100:
                # np.set_printoptions(threshold=99999, linewidth=2000)
                # print(np.array2string(cls_mask ,max_line_width=2000,separator=''))
            
            # gauss_sum[img,cls] += np.sum(pdf_arr[class_indices],axis=0)[0]
            gauss_sum[img,cls] = np.multiply(pdf_sum_abs, cls_mask) 

        # print('    shape of pdf_array_list : ' , len(pdf_class_list))
        gauss_scatt = np.stack(pdf_class_list, axis = 0)
        # print('       pdf_class.shape : ' ,gauss_scatt.shape)         
        pdf_batch_list.append(gauss_scatt)
        
    # print(gauss_sum)
    # if rois_per_image == 100:
            # print('cls_mask[0,0]\n',cls_mask[0,0])
            # print('cls_mask[0,1]\n',cls_mask[0,1])
            # print('cls_mask[0,2]\n',cls_mask[0,2])
            # print('cls_mask[0,3]\n',cls_mask[0,3])
            # print('cls_mask[1,0]\n',cls_mask[1,0])
            # print('cls_mask[1,1]\n',cls_mask[1,1])
            # print('cls_mask[1,2]\n',cls_mask[1,2])
            # print('cls_mask[1,3]\n',cls_mask[1,3])
            
    gauss_sum = np.where(gauss_sum > 1e-6, gauss_sum,0)
    # gauss_scatt = np.stack(pdf_batch_list, axis = 0)                
    # means_arr = np.stack(means_list, axis = 0)
    # covar_arr = np.stack(covar_list, axis = 0)
    print('   gauss_sum shape            : ',gauss_sum.shape)
    gauss_sum = np.transpose(gauss_sum, [0,2,3,1])
    # print('   Final gauss_scatt.shape : ' ,gauss_scatt.shape)         
    # print('   covar_arr shape : ', covar_arr.shape, '  means_arr shape', means_arr.shape)
    # print('   pdf_arr : ', pdf_arr.dtype)
    return  gauss_sum  #  [ gauss_sum, gauss_scatt, means_arr, covar_arr]   
    

class PCNLayer(KE.Layer):
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
        print('\n>>> PCN Layer ')
        self.config = config

        
    def call(self, inputs):
        
        print('>>> PCN Layer : call ', type(inputs), len(inputs))
        print('     mrcnn_class.shape    :',  inputs[0].shape, type(inputs[0]))
        print('     mrcnn_bbox.shape     :',  inputs[1].shape, type(inputs[1])) 
        print('     output_rois.shape    :',  inputs[3].shape, type(inputs[3])) 
        # print('     mrcnn_mask.shape     :',  inputs[2].shape, type(inputs[2])) 
        # print('     gt_class_ids.shape   :',  inputs[4].shape, type(inputs[4])) 
        # print('     gt_bboxes.shape      :',  inputs[5].shape, type(inputs[5])) 
        mrcnn_class, mrcnn_bbox,  output_rois, gt_class_ids, gt_bboxes = inputs
        

        def wrapper(mrcnn_class, mrcnn_bbox, output_rois, gt_class_ids, gt_bboxes):
            # print('>>> PCN Layer Wrapper: call')

            pred_tensor, pred_cls_cnt = build_predictions(mrcnn_class, mrcnn_bbox,  output_rois, self.config)
            # print('   pred_tensor : ', pred_tensor.shape, 'pred_cls_cnt: ', pred_cls_cnt.shape)

            gt_tensor  , gt_cls_cnt = build_ground_truth(gt_class_ids, gt_bboxes, self.config)
            # print('   gt_tensor  : ', gt_tensor.shape    , ' gt_cls_cnt : ', gt_cls_cnt.shape)
           
            # print('   Build Gaussian np for detected rois =========================')    
            # pred_gaussian, pred_scatter, pred_means, pred_covar = build_gaussian_np(pred_tensor, pred_cls_cnt, self.config)
            pred_gaussian = build_gaussian_np(pred_tensor, pred_cls_cnt, self.config)
            # print('   Output build_gaussian_tf (predicitons)')
            # print('   Pred scatter : ', pred_scatter.shape, '   pred gaussian : ', pred_gaussian.shape)
            # print('   Pred means   : ', means.shape       , '   pred covar    : ', covar.shape)
            
            # print('   Build Gaussian np for ground_truth ==========================')    
            # gt_gaussian, gt_scatter, gt_means, gt_covar  = build_gaussian_np(gt_tensor , gt_cls_cnt, self.config)
            gt_gaussian   = build_gaussian_np(gt_tensor , gt_cls_cnt, self.config)
            # print('   gt scatter : ', gt_scatter.shape, '   gt gaussian : ', gt_gaussian.shape)
            # print('   gt means   : ', gt_means.shape  , '   gt covar    : ', gt_covar.shape)
            
            # # Stack detections and cast to float32
            # # TODO: track where float64 is introduced

            return  [ 
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

        return tf.py_func(wrapper, inputs, [  
                                               tf.float32                 # pred_gaussian
                                               # , tf.float64               # pred_scatter
                                               # , tf.float32               # pred_means
                                               # , tf.float32               # pred_covar
                                                , tf.float32                # pred_tensor 
                                                , tf.int16                  # pcs_cls_cnt
                                                , tf.float32                # gt_gaussian
                                                # , tf.float64                # gt_scatter
                                                # , tf.float32                # gt_means 
                                                # , tf.float32                # gt_covar
                                                , tf.float32                # gt_tensor
                                                , tf.int16                  # gt_cls_cnt     
                                           ])
        

        
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        return [
           (None, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1], self.config.NUM_CLASSES)    # pred_gaussian 
           # , (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1])   
           # , (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, 2)        # means  
           # , (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, 2)        # covar  
           , (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, 8)        # pred_tensors 
           , (None, self.config.NUM_CLASSES)                                             # pred_cls_cnt

           , (None, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1], self.config.DETECTION_MAX_INSTANCES) # GT_GAUSSIAN
           # , (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[1])

           # , (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, 2)   # gt_means                         
           # , (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, 2)   # gt_covar                         
            
           , (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES, 8)    # gt_tensors             
           , (None, self.config.NUM_CLASSES)                                            # gt_cls_cnt
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
        
            pred_tensor, pred_cls_cnt , gt_tensor, gt_cls_cnt = \
                    build_predictions(mrcnn_class, mrcnn_bbox, output_rois, self.config)
            # print('pred_tensor : ', pred_tensor.shape)
            # print(pred_tensor)
            # print('pred_cls_cnt: ', pred_cls_cnt.shape)
            # print(pred_cls_cnt)
            print(' Build Gaussian np for detected rois =========================')    
            pred_gaussian = build_gaussian_np(pred_tensor, pred_cls_cnt, self.config)

            return pred_gaussian, pred_tensor, pred_cls_cnt

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

