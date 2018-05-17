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
from collections import OrderedDict
import numpy as np
import pprint
import scipy.misc
import tensorflow as tf

import keras
import keras.backend as KB
import keras.layers  as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
#sys.path.append('..')

import mrcnn.utils            as utils
import mrcnn.loss             as loss
from   mrcnn.datagen          import data_generator
from   mrcnn.utils            import log
from   mrcnn.utils            import parse_image_meta_graph, parse_image_meta

from   mrcnn.RPN_model        import build_rpn_model
from   mrcnn.resnet_model     import resnet_graph

from   mrcnn.chm_layer        import CHMLayer, CHMLayerInference
from   mrcnn.proposal_layer   import ProposalLayer

# from   mrcnn.fcn_layer        import fcn_graph 
from   mrcnn.fcn_layer        import fcn_graph
# from   mrcnn.detect_layer     import DetectionLayer  
from   mrcnn.detect_tgt_layer_mod import DetectionTargetLayer_mod

from   mrcnn.fpn_layers       import fpn_graph, fpn_classifier_graph, fpn_mask_graph
from   mrcnn.callbacks        import MyCallback
from   mrcnn.batchnorm_layer  import BatchNorm

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3.0")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')
pp = pprint.PrettyPrinter(indent=2, width=100)
tf.get_variable_scope().reuse_variables()

############################################################
#  Code                                     moved to 
#  -----------------------------------      ----------------
#  BatchNorm                              batchnorm_layer.py      
#  Miscellenous Graph Functions                     utils.py 
#  Loss Functions                                    loss.py
#  Data Generator                                 datagen.py
#  Data Formatting                                  utils.py
#  Proposal Layer                          proposal_layer.py
#  ROIAlign Layer                         roiialign_layer.py
#  FPN Layers                                   fpn_layer.py
#  FPN Head Layers                         fpnhead_layers.py
#  Detection Target Layer                detect_tgt_layer.py
#  Detection Layer                        detection_layer.py
#  Region Proposal Network (RPN)                rpn_model.py
#  Resnet Graph                              resnet_model.py
############################################################

############################################################
##  MaskRCNN Class
############################################################
class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']

        print('>>> Initialize model WITHOUT MASKING LAYERS!!!!')

        self.mode      = mode
        self.config    = config
        self.model_dir = model_dir
        self.set_log_dir()
        # Pre-defined layer regular expressions
        self.layer_regex = {
            # ResNet from a specific stage and up
            "res3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            "res4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            "res5+": r"(res5.*)|(bn5.*)",

            # fcn only 
            "fcn" : r"(fcn\_.*)",
            # fpn
            "fpn" : r"(fpn\_.*)",
            # rpn
            "rpn" : r"(rpn\_.*)",
            # rpn
            "mrcnn" : r"(mrcnn\_.*)",

            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # all layers but the backbone
            "allheads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(fcn\_.*)",
          
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        

        # self.keras_model = self.build(mode=mode, config=config)
        self.keras_model = self.build_mod(mode=mode, config=config)

        print('>>> MODIFIED MaskRCNN initialization complete -- WITHOUT MASKING LAYERS!!!!')

    
    def build_mod(self, mode, config):
        '''
        Build MODIFIED Mask R-CNN architecture (NO MASK PROCESSING)
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        '''
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        ##------------------------------------------------------------------                            
        ##  Input Layer
        ##------------------------------------------------------------------
        input_image      = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")
        
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox  = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            
            # Normalize coordinates
            h, w = KB.shape(input_image)[1], KB.shape(input_image)[2]
            image_scale = KB.cast(KB.stack([h, w, h, w], axis=0), tf.float32)
            input_normlzd_gt_boxes = KL.Lambda(lambda x: x / image_scale)(input_gt_boxes)
            
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            # If using USE_MINI_MASK the mask is 56 x 56 x None 
            #    else:    image h x w x None
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        # End if mode == 'training'
        
        ##----------------------------------------------------------------------------
        ##  Resnet Backbone
        ##----------------------------------------------------------------------------
        # Build the Resnet shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        #----------------------------------------------------------------------------
        Resnet_Layers      = resnet_graph(input_image, "resnet50", stage5=True)
        
        ##----------------------------------------------------------------------------
        ##  FPN network - Build the Feature Pyramid Network (FPN) layers.
        ##----------------------------------------------------------------------------
        P2, P3, P4, P5, P6 = fpn_graph(Resnet_Layers)
        
        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps   = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        ##----------------------------------------------------------------------------        
        ##  Generate Anchors Box Coordinates
        ##  shape.anchors will contain an array of anchor box coordinates (y1,x1,y2,x2)
        ##----------------------------------------------------------------------------        
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                      config.RPN_ANCHOR_RATIOS,
                                                      config.BACKBONE_SHAPES,
                                                      config.BACKBONE_STRIDES,
                                                      config.RPN_ANCHOR_STRIDE)


        ##----------------------------------------------------------------------------        
        ##  RPN Model - 
        ##  model which is applied on the feature maps produced by the resnet backbone
        ##----------------------------------------------------------------------------                
        RPN_model = build_rpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), 256)

        
        #----------------------------------------------------------------------------         
        # Loop through pyramid layers (P2 ~ P6) and pass each layer to the RPN network
        # for each layer rpn network returns [rpn_class_logits, rpn_probs, rpn_bbox]
        #----------------------------------------------------------------------------
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(RPN_model([p]))

            
        #----------------------------------------------------------------------------                    
        # Concatenate  layer outputs
        #----------------------------------------------------------------------------                        
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        # 
        # the final output is a list consisting of three tensors:
        #
        # a1,..., a5: rpn_class_logits : Tensor("rpn_class_logits, shape=(?, ?, 2), dtype=float32)
        # b1,..., b5: rpn_probs        : Tensor("rpn_class       , shape=(?, ?, 2), dtype=float32)
        # c1,..., c5: rpn_bbox         : Tensor("rpn_bbox_11     , shape=(?, ?, 4), dtype=float32)
        #----------------------------------------------------------------------------                
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]     
        outputs = list(zip(*layer_outputs))
        # concatinate the list of tensors in each group (logits, probs, bboxes)
                
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))  for o, n in zip(outputs, output_names)]
        print('\n>>> RPN Outputs ',  type(outputs))
        for i in outputs:
            print('     ', i.name)
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        ##----------------------------------------------------------------------------                
        ##  RPN Proposal Layer
        ##----------------------------------------------------------------------------                
        # Generate proposals from bboxes and classes genrated by the RPN model
        # Proposals are [batch, proposal_count, 4 (y1, x1, y2, x2)] in NORMALIZED coordinates
        # and zero padded.
        #
        # proposal_count : number of proposal regions to generate:
        #   Training  mode :        2000 proposals 
        #   Inference mode :        1000 proposals
        #----------------------------------------------------------------------------                        
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
                    else config.POST_NMS_ROIS_INFERENCE

        rpn_proposal_rois = ProposalLayer(proposal_count=proposal_count,            # num proposals to generate
                                 nms_threshold=config.RPN_NMS_THRESHOLD,
                                 name="rpn_proposal_rois",
                                 anchors=self.anchors,
                                 config=config)([rpn_class, rpn_bbox])
                                 
        ##----------------------------------------------------------------------------                
        ## Training Mode Layers
        ##----------------------------------------------------------------------------                
        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image came from. 
            _, _, _, active_class_ids = KL.Lambda(lambda x:  parse_image_meta_graph(x), mask=[None, None, None, None])(input_image_meta)
            
            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs - Normalize and use ROIs provided as an input.
                input_rois  = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4], name="input_roi", dtype=np.int32)
                rpn_proposal_rois = KL.Lambda(lambda x: KB.cast(x, tf.float32) / image_scale[:4], name='rpn_proposal_rois')(input_rois)
            else:
                pass
                # target_rois = rpn_proposal_rois
            
            ##--------------------------------------------------------------------------------------
            ##  DetetcionTargetLayer
            ##--------------------------------------------------------------------------------------
            #  Generate detection targets
            #    generated RPNs ----> Target ROIs
            #
            #    target_* returned from this layer are the 'processed' versions of gt_*  
            # 
            #    Subsamples proposals and generates target outputs for training
            #    Note that proposal class IDs, input_normalized_gt_boxes, and gt_masks are zero padded. 
            #    Equally, returned rois and targets are zero padded.
            # 
            #   Note : roi (first output of DetectionTargetLayer) was testing and verified to b
            #          be equal to output_rois. Therefore, the output_rois layer was removed, 
            #          and the first output below was renamed rois --> output_rois
            #   
            #    output_rois :       (?, TRAIN_ROIS_PER_IMAGE, 4),    # output bounindg boxes            
            #    target_class_ids :  (?, 1),                          # gt class_ids            
            #    target_bbox_deltas: (?, TRAIN_ROIS_PER_IMAGE, 4),    # gt bounding box deltas            
            #    roi_gt_bboxes:      (?, TRAIN_ROIS_PER_IMAGE, 4)     # gt bboxes            
            #
            #--------------------------------------------------------------------------------------
            # remove target_mask for build_new   05-11-2018
            
            output_rois, target_class_ids, target_bbox_deltas,  roi_gt_boxes = \
                DetectionTargetLayer_mod(config, name="proposal_targets") \
                                    ([rpn_proposal_rois, input_gt_class_ids, input_normlzd_gt_boxes])

            #--------------------------------------------------------------------------------------
            # TODO: clean up (use tf.identify if necessary)
            # replace with KB.identity -- 03-05-2018
            # renamed output from DetectionTargetLayer abouve from roi to output_roi and 
            # following lines were removed. 
            # 
            # output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)
            # output_rois = KL.Lambda(lambda x: KB.identity(x), name= "output_rois")(rois)
            #------------------------------------------------------------------------------------

            ##------------------------------------------------------------------------------------
            ##  MRCNN Network Classification Head
            ##  TODO: verify that this handles zero padded ROIs
            ##------------------------------------------------------------------------------------
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                fpn_classifier_graph(output_rois, mrcnn_feature_maps, config.IMAGE_SHAPE, config.POOL_SIZE, config.NUM_CLASSES)

            #------------------------------------------------------------------------------------
            #  MRCNN Network Mask Head
            #------------------------------------------------------------------------------------
            # mrcnn_mask = \
                # fpn_mask_graph(output_rois, mrcnn_feature_maps, config.IMAGE_SHAPE, config.MASK_POOL_SIZE, config.NUM_CLASSES)



            ##----------------------------------------------------------------------------
            ##  Contextual Layer(CHM) to generate contextual feature maps using outputs from MRCNN 
            ##----------------------------------------------------------------------------         
            # Once we are comfortable with the results we can remove additional outputs from here.....
            pred_heatmap , gt_heatmap, pred_heatmap_norm , gt_heatmap_norm, pred_tensor , gt_tensor, gt_deltas    \
               =  CHMLayer(config, name = 'cntxt_layer' ) \
                    ([mrcnn_class, mrcnn_bbox, output_rois, input_gt_class_ids, input_gt_boxes, target_class_ids,target_bbox_deltas])
            print('<<<  shape of pred_heatmap   : ', pred_heatmap.shape, ' Keras tensor ', KB.is_keras_tensor(pred_heatmap) )                         
            print('<<<  shape of gt_heatmap     : ', gt_heatmap.shape  , ' Keras tensor ', KB.is_keras_tensor(gt_heatmap) )
            
                         
            ##------------------------------------------------------------------------
            ##  FCN Network Head
            ##------------------------------------------------------------------------
            # fcn_heatmap, fcn_class_logits, fcn_scores, fcn_bbox_deltas = fcn_graph_mod(pred_heatmap, config)
            # fcn_heatmap = fcn_graph(pred_heatmap, config)
            # print('   fcn_heatmap  shape is : ', KB.int_shape(fcn_heatmap), ' Keras tensor ', KB.is_keras_tensor(fcn_heatmap) )        

            ##------------------------------------------------------------------------
            ##  Loss layer definitions
            ##------------------------------------------------------------------------
            print('\n')
            print('---------------------------------------------------')
            print('    building Loss Functions ')
            print('---------------------------------------------------')
            print(' gt_deltas         :', KB.is_keras_tensor(gt_deltas)      , KB.int_shape(gt_deltas        ))
            # print(' fcn_bbox_deltas   :', KB.is_keras_tensor(fcn_bbox_deltas), KB.int_shape(fcn_bbox_deltas  ))
            print(' target_class_ids  :', KB.is_keras_tensor(target_class_ids), KB.int_shape(target_class_ids ))

            rpn_class_loss   = KL.Lambda(lambda x: loss.rpn_class_loss_graph(*x),        name="rpn_class_loss")\
                                 ([input_rpn_match   , rpn_class_logits])
            
            rpn_bbox_loss    = KL.Lambda(lambda x: loss.rpn_bbox_loss_graph(config, *x),  name="rpn_bbox_loss")\
                                 ([input_rpn_bbox    , input_rpn_match   , rpn_bbox])

            mrcnn_class_loss = KL.Lambda(lambda x: loss.mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")\
                                 ([target_class_ids  , mrcnn_class_logits, active_class_ids])
            
            mrcnn_bbox_loss  = KL.Lambda(lambda x: loss.mrcnn_bbox_loss_graph(*x),  name="mrcnn_bbox_loss") \
                                 ([target_bbox_deltas, target_class_ids  , mrcnn_bbox])

            # mrcnn_mask_loss  = KL.Lambda(lambda x: loss.mrcnn_mask_loss_graph(*x),  name="mrcnn_mask_loss") \
                                # ([target_mask       , target_class_ids  , mrcnn_mask])

            # fcn_bbox_loss    = KL.Lambda(lambda x: loss.fcn_bbox_loss_graph(*x),  name="fcn_bbox_loss") \
                                 # ([gt_deltas, target_class_ids  , fcn_bbox_deltas])
                                
            # fcn_norm_loss  = KL.Lambda(lambda x: loss.fcn_norm_loss_graph(*x),  name="fcn_norm_loss") \
                             # ([gt_heatmap, fcn_heatmap])
            # print('\n\n\n')
            # print('---------------------------------------------------')
            # print('    building fcn_norm_loss')
            # print('---------------------------------------------------')
            # fcn_loss       = KL.Lambda(lambda x: loss.fcn_loss_graph(*x), name="fcn_loss") \
                            # ([gt_heatmap, fcn_heatmap])
            
            # print('\n Keras Tensors?? ')
            print(' output_rois       :', KB.is_keras_tensor(output_rois ))
            print(' pred_heatmap      :', KB.is_keras_tensor(pred_heatmap))
            print(' gt_heatmap        :', KB.is_keras_tensor(gt_heatmap))
            # print(' pred_cls_cnt     :', KB.is_keras_tensor(pred_cls_cnt2))
            # print(' mask_loss         :', KB.is_keras_tensor(mrcnn_mask_loss))
            # print(' rpn_proposal_rois :', KB.is_keras_tensor(rpn_proposal_rois))
            
            # print(' fcn_heatmap       :', KB.is_keras_tensor(fcn_heatmap))
            # print(' fcn_class_logits  :', KB.is_keras_tensor(fcn_class_logits))
            # print(' fcn_scores        :', KB.is_keras_tensor(fcn_scores))
            print(' gt_deltas         :', KB.is_keras_tensor(gt_deltas))
            # print(' fcn_bbox_deltas   :', KB.is_keras_tensor(fcn_bbox_deltas))
            # print(' target_class_ids  :', KB.is_keras_tensor(fcn_bbox_deltas))
            # print(' fcn_bbox_loss     :', KB.is_keras_tensor(fcn_loss))
            # print(' fcn_norm_loss     :', KB.is_keras_tensor(fcn_norm_loss))


            # Model Inputs 
            inputs = [ 
                       input_image,              #  
                       input_image_meta,         #   
                       input_rpn_match ,         # [batch_sz, N, 1:<pos,neg,nutral>)                  [ 1,4092, 1]
                       input_rpn_bbox  ,         # [batch_sz, RPN_TRAIN_ANCHORS_PER_IMAGE, 4]         [ 1, 256, 4]
                       input_gt_class_ids,       # [batch_sz, MAX_GT_INSTANCES] Integer class IDs         [1, 100]
                       input_gt_boxes,           # [batch_sz, MAX_GT_INSTANCES, 4]                     [1, 100, 4]
                       input_gt_masks            # [batch_sz, height, width, MAX_GT_INSTANCES].   [1, 56, 56, 100]
                     ]
                        
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)

            outputs =  [   rpn_class_logits   , rpn_class         , rpn_bbox            , rpn_proposal_rois                                             # 3
                         , output_rois        , target_class_ids  , target_bbox_deltas  , roi_gt_boxes  # 4 -8    
                         , mrcnn_class_logits , mrcnn_class       , mrcnn_bbox                             # 9 -  12 (from FPN)
                         , rpn_class_loss     , rpn_bbox_loss                                               # 13 - 14
                         , mrcnn_class_loss   , mrcnn_bbox_loss                                             # 15 - 17
                         # , fcn_bbox_loss 
                         , pred_heatmap                                                                    # 18
                         , gt_heatmap                                                                      # 19
                         , pred_heatmap_norm                                                                    # 18
                         , gt_heatmap_norm                                                                      # 19
                         , pred_tensor                                                                      # 20
                         , gt_tensor                                                                        # 21    
                         , gt_deltas                                                                        # 22    
              
                         ]
                         # , active_class_ids
                         # , fcn_loss
                         # , fcn_heatmap                                                                      # 12
        
        # end if Training
        
        
        
        model = KM.Model( inputs, outputs,  name='mask_rcnn')
        
        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        print('\n>>> MODIFIED MaskRCNN build complete -- WITHOUT MASKING LAYERS!!!!')
        return model

        
##-------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------        
##-------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------        
        
        
        
                
    def detect(self, images, verbose=0):
        '''
        Runs the detection pipeline.

        images:         List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois:           [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids:      [N] int class IDs
        scores:         [N] float probability scores for the class IDs
        masks:          [H, W, N] instance binary masks
        '''
        # print('>>> model detect()')
        
        assert self.mode   == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)
                
        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)
        if verbose:
            log("molded_images", molded_images)
            log("image_metas"  , image_metas)
            
        ## Run object detection pipeline
        # print('    call predict()')
        detections, rpn_proposal_rois, rpn_class, rpn_bbox,\
                    mrcnn_class, mrcnn_bbox, mrcnn_mask \
                              =  self.keras_model.predict([molded_images, image_metas], verbose=0)
            
        # print('    return from  predict()')
        # print('    Length of detections : ', len(detections))
        # print('    Length of rpn_proposal_rois   : ', len(rpn_proposal_rois   ))
        # print('    Length of rpn_class  : ', len(rpn_class  ))
        # print('    Length of rpn_bbox   : ', len(rpn_bbox   ))
        # print('    Length of mrcnn_class: ', len(mrcnn_class))
        # print('    Length of mrcnn_bbox : ', len(mrcnn_bbox ))
        # print('    Length of mrcnn_mask : ', len(mrcnn_mask ))

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], 
                                       mrcnn_mask[i],
                                       image.shape  ,
                                       windows[i])
            results.append({
                "rois"     : final_rois,
                "class_ids": final_class_ids,
                "scores"   : final_scores,
                "masks"    : final_masks,
            })
        return results


        
    def find_last(self):
        """
        Finds the last checkpoint file of the last trained model in the
        model directory.
        
        Returns:
        --------
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        print('>>> find_last checkpoint in : ', self.model_dir)
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        # log("    find_last info:   dir_name: {}".format(dir_name))
        # log("    find_last info: checkpoint: {}".format(checkpoint))
        return dir_name, checkpoint

    def save_model(self, filepath, by_name=False, exclude=None):
        """
        Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        print('>>> save_model_architecture()')

        model_json = self.keras_model.to_json()
        full_filepath = os.path.join(filepath, filename)
        log('    save model to  {}'.format(full_filepath))

        with open(full_filepath , 'w') as f:
            # json.dump(model_json, full_filepath)               
            if hasattr(f, 'close'):
                f.close()
                print('file closed')
                
                
        print('    save_weights: save directory is  : {}'.format(filepath))
        print('    save model Load weights complete')        
        return(filepath)

        
    def load_weights(self, filepath, by_name=False, exclude=None):
        """
        Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology
        print('>>> load_weights()')
        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        
        log('    load_weights: Loading weights from: {}'.format(filepath))
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)
       
        # print(' layers to load ' )
        # print('----------------' )
        # for idx,layer in enumerate(layers):
            # print('>layer {} : name : {:40s}  type: {}'.format(idx,layer.name,layer))

            
        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()
        
        log('    load_weights: Log directory set to : {}'.format(filepath))
        # Update the log directory
        self.set_log_dir(filepath)
        print('    Load weights complete : ',filepath)        
        return(filepath)

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        # print('>>> Set_log_dir() -- model dir is ', self.model_dir)
        # print('    model_path           :   ', model_path)
        # print('    config.LAST_EPOCH_RAN:   ', self.config.LAST_EPOCH_RAN)

        self.tb_dir = os.path.join(self.model_dir,'tensorboard')
        # self.epoch  = 0
        last_checkpoint_epoch = 0
        now = datetime.datetime.now()
        
        # If we have a model path with date and epochs use them
        
        if model_path:
            # Continue from we left off. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            model_path = model_path.replace('\\' , "/")
            # print('    set_log_dir: model_path (input) is : {}  '.format(model_path))        

            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:             
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                last_checkpoint_epoch = int(m.group(6)) + 1
            # print('    set_log_dir: self.epoch set to {}  (Next epoch to run)'.format(self.epoch))
            # print('    set_log_dir: tensorboard path: {}'.format(self.tb_dir))

        if last_checkpoint_epoch > 0 and  self.config.LAST_EPOCH_RAN > last_checkpoint_epoch: 
            self.epoch = self.config.LAST_EPOCH_RAN
        else :
            self.epoch = last_checkpoint_epoch
        
        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
                self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")
        log('    set_log_dir: Checkpoint path set to : {}'.format(self.checkpoint_path))
        log('    set_log_dir: self.epoch set to {} '.format(self.epoch))

        
        
    def get_imagenet_weights(self):
        """
        Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path


    def mold_inputs(self, images):
        '''
        Takes a list of images and modifies them to the format expected as an 
        input to the neural network.
        
        - resize to IMAGE_MIN_DIM / IMAGE_MAX_DIM  : utils.RESIZE_IMAGE()
        - subtract mean pixel vals from image pxls : utils.MOLD_IMAGE()
        - build numpy array of image metadata      : utils.COMPOSE_IMAGE_META()
        
        images       :     List of image matricies [height,width,depth]. Images can have
                           different sizes.

        Returns 3 Numpy matrices:
        
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas  : [N, length of meta data]. Details about each image.
        windows      : [N, (y1, x1, y2, x2)]. The portion of the image that has the
                       original image (padding excluded).
        '''
        
        molded_images = []
        image_metas   = []
        windows       = []
        
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            
            # subtract mean pixel values from image pixels
            molded_image = utils.mold_image(molded_image, self.config)
            
            # Build image_meta
            image_meta = utils.compose_image_meta( 0, 
                                                   image.shape, 
                                                   window,
                                                   np.zeros([self.config.NUM_CLASSES],
                                                   dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            image_metas.append(image_meta)
            windows.append(window)
        
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas   = np.stack(image_metas)
        windows       = np.stack(windows)
        return molded_images, image_metas, windows

        
    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        '''
        Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the application.

        detections  : [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask  : [N, height, width, num_classes]
        image_shape : [height, width, depth] Original size of the image before resizing
        window      : [y1, x1, y2, x2] Box in the image where the real image is
                        excluding the padding.

        Returns:
        boxes       : [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids   : [N] Integer class IDs for each bounding box
        scores      : [N] Float probability scores of the class_id
        masks       : [height, width, num_instances] Instance masks
        '''
        
        # print('>>>  unmold_detections ')
        # print('     detections.shape : ', detections.shape)
        # print('     mrcnn_mask.shape : ', mrcnn_mask.shape)
        # print('     image_shape.shape: ', image_shape)
        # print('     window.shape     : ', window)
        # print(detections)
        
        # How many detections do we have?
        # Detections array is padded with zeros. detections[:,4] identifies the class 
        # Find all rows in detection array with class_id == 0 , and place their row indices
        # into zero_ix. zero_ix[0] will identify the first row with class_id == 0.
        
        np.set_printoptions(linewidth=100)        

        zero_ix = np.where(detections[:, 4] == 0)[0]
    
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # print(' np.where() \n', np.where(detections[:, 4] == 0))
        # print('     zero_ix.shape     : ', zero_ix.shape)
        # print('     N is :', N)
        
        # Extract boxes, class_ids, scores, and class-specific masks
        boxes     = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores    = detections[:N, 5]
        masks     = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty((0,) + masks.shape[1:3])

        return boxes, class_ids, scores, full_masks

        
    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

        
    def run_graph(self, images, outputs):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(KB.learning_phase(), int):
            inputs += [KB.learning_phase()]
        kf = KB.function(model.inputs, list(outputs.values()))

        # Run inference
        molded_images, image_metas, windows = self.mold_inputs(images)
        # TODO: support training mode?
        # if TEST_MODE == "training":
        #     model_in = [molded_images, image_metas,
        #                 target_rpn_match, target_rpn_bbox,
        #                 input_normalized_gt_boxes, gt_masks]
        #     if not config.USE_RPN_ROIS:
        #         model_in.append(target_rois)
        #     if model.uses_learning_phase and not isinstance(KB.learning_phase(), int):
        #         model_in.append(1.)
        #     outputs_np = kf(model_in)
        # else:

        model_in = [molded_images, image_metas]
        if model.uses_learning_phase and not isinstance(KB.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np

      
    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

        
    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
            else:
                print('   Layer: ', l.name, ' doesn''t have any weights !!!')
        return layers

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        '''
        Sets model layers as trainable if their names match the given
        regular expression.
        '''       
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("\nSelecting layers to train")
            log("-------------------------")
            log("{:5}    {:20}     {}".format( 'Layer', 'Layer Name', 'Layer Type'))

        keras_model = keras_model or self.keras_model
              
      
        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # go through layers one by one, if the layer matches a layer reg_ex, set it to trainable 
        for ind,layer in enumerate(layers):
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                if verbose > 0:
                    print("Entering model layer: ", layer.name, '------------------------------')
                
                self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
                indent -= 4
                
                if verbose >  0 :
                    print("Exiting model layer ", layer.name, '--------------------------------')
                continue

            if not layer.weights:
                if verbose > 0:
                    log(" {}{:3}  {:20}   ({:20})   ............................no weights to train ]". \
                    format(" " * indent, ind, layer.name,layer.__class__.__name__))
                continue
                
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))

            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names

            if verbose > 0:
                if trainable :
                    log(" {}{:3}  {:20}   ({:20})   TRAIN ".\
                        format(" " * indent, ind, layer.name, layer.__class__.__name__))
                else:
                    log(" {}{:3}  {:20}   ({:20})   ............................not a layer we want to train ]". \
                        format(" " * indent, ind, layer.name, layer.__class__.__name__))                
                    pass
        return   
        
           
    def train(self, 
              train_dataset, val_dataset, 
              learning_rate, 
              layers            = None,
              losses            = None,
              epochs            = 0,
              epochs_to_run     = 0,
              batch_size        = 0, 
              steps_per_epoch   = 0,
              min_LR            = 0.00001):
        '''
        Train the model.
        train_dataset, 
        val_dataset:    Training and validation Dataset objects.
        
        learning_rate:  The learning rate to train with
        
        layers:         Allows selecting wich layers to train. It can be:
                        - A regular expression to match layer names to train
                        - One of these predefined values:
                        heads: The RPN, classifier and mask heads of the network
                        all: All the layers
                        3+: Train Resnet stage 3 and up
                        4+: Train Resnet stage 4 and up
                        5+: Train Resnet stage 5 and up
        
        losses:         List of losses to monitor.
        
        epochs:         Number of training epochs. Note that previous training epochs
                        are considered to be done already, so this actually determines
                        the epochs to train in total rather than in this particaular
                        call.
        
        epochs_to_run:  Number of epochs to run, will update the 'epochs parm.                        
                        
        '''
        assert self.mode == "training", "Create model in training mode."
        
        if batch_size == 0 :
            batch_size = self.config.BATCH_SIZE
        if epochs_to_run > 0 :
            epochs = self.epoch + epochs_to_run
        if steps_per_epoch == 0:
            steps_per_epoch = self.config.STEPS_PER_EPOCH
            
        # use Pre-defined layer regular expressions
        # if layers in self.layer_regex.keys():
            # layers = self.layer_regex[layers]
        print(layers)
        # train_regex_list = []
        # for x in layers:
            # print( ' layers ias : ',x)
            # train_regex_list.append(x)
        train_regex_list = [self.layer_regex[x] for x in layers]
        print(train_regex_list)
        layers = '|'.join(train_regex_list)        
        print('layers regex :', layers)
        

            
        
        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=batch_size)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                        batch_size=batch_size,
                                        augment=False)

        # my_callback = MyCallback()

        # Callbacks
        ## call back for model checkpoint was originally (?) loss. chanegd to val_loss (which is default) 2-5-18
        callbacks = [
              keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                          histogram_freq=0,
                                          batch_size=32,
                                          write_graph=True,
                                          write_grads=False,
                                          write_images=True,
                                          embeddings_freq=0,
                                          embeddings_layer_names=None,
                                          embeddings_metadata=None)

            , keras.callbacks.ModelCheckpoint(self.checkpoint_path, 
                                              mode = 'auto', 
                                              period = 1, 
                                              monitor='val_loss', 
                                              verbose=1, 
                                              save_best_only = True, 
                                              save_weights_only=True)
                                            
            , keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                mode     = 'auto', 
                                                factor   = 0.3, 
                                                cooldown = 50, 
                                                patience = 150, 
                                                min_lr   = min_LR, 
                                                verbose  = 1)                                            
                                                
            , keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                mode      = 'auto', 
                                                min_delta = 0.00001, 
                                                patience  = 200, 
                                                verbose   = 1)                                            
        ]

        # Train

        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM, losses)
        
        log("Starting at epoch {} of {} epochs. LR={}\n".format(self.epoch, epochs, learning_rate))
        log("Steps per epochs {} ".format(steps_per_epoch))
        log("Batch size       {} ".format(batch_size))
        log("Checkpoint Path: {} ".format(self.checkpoint_path))
        
        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            validation_data=next(val_generator),
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=1,                                  # max(self.config.BATCH_SIZE // 2, 2),
            use_multiprocessing=False
        )
        self.epoch = max(self.epoch, epochs)

        print('Final : self.epoch {}   epochs {}'.format(self.epoch, epochs))

        
        
        
    def train_in_batches(self, 
              train_dataset, val_dataset, 
              learning_rate, 
              layers,
              losses            = None,              
              epochs_to_run     = 1, 
              batch_size        = 0, 
              steps_per_epoch   = 0):
        '''
        Train the model.
        train_dataset, 
        val_dataset:    Training and validation Dataset objects.
        
        learning_rate:  The learning rate to train with
        
        epochs:         Number of training epochs. Note that previous training epochs
                        are considered to be done already, so this actually determines
                        the epochs to train in total rather than in this particaular
                        call.
                        
        layers:         Allows selecting wich layers to train. It can be:
                        - A regular expression to match layer names to train
                        - One of these predefined values:
                        heads: The RPN, classifier and mask heads of the network
                        all: All the layers
                        3+: Train Resnet stage 3 and up
                        4+: Train Resnet stage 4 and up
                        5+: Train Resnet stage 5 and up
        '''
        assert self.mode == "training", "Create model in training mode."
        
       
        # Use Pre-defined layer regular expressions
        # if layers in self.layer_regex.keys():
            # layers = self.layer_regex[layers]
        print(layers)
        train_regex_list = [self.layer_regex[x] for x in layers]
        print(train_regex_list)
        layers = '|'.join(train_regex_list)        
        print('layers regex :', layers)
        
            
        if batch_size == 0 :
            batch_size = self.config.BATCH_SIZE            
            
        if steps_per_epoch == 0:
            steps_per_epoch = self.config.STEPS_PER_EPOCH
            

        
        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=batch_size)
        val_generator   = data_generator(val_dataset, self.config, shuffle=True,
                                         batch_size=batch_size,
                                         augment=False)
       
        log("    Last epoch completed : {} ".format(self.epoch))
        log("    Starting from epoch  : {} for {} epochs".format(self.epoch, epochs_to_run))
        log("    Learning Rate        : {} ".format(learning_rate))
        log("    Steps per epoch      : {} ".format(steps_per_epoch))
        log("    Batchsize            : {} ".format(batch_size))
        log("    Checkpoint Folder    : {} ".format(self.checkpoint_path))
        epochs = self.epoch + epochs_to_run
        
        from tensorflow.python.platform import gfile
        if not gfile.IsDirectory(self.log_dir):
            log('Creating checkpoint folder')
            gfile.MakeDirs(self.log_dir)
        else:
            log('Checkpoint folder already exists')
        
        self.set_trainable(layers)            
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM, losses)        
        
        # copied from \keras\engine\training.py
        # def _get_deduped_metrics_names(self):
        ## get metrics from keras_model.metrics_names
        out_labels = self.get_deduped_metrics_names()
        print(' ====> out_labels : ', out_labels)

        ## setup Progress Bar callback
        callback_metrics = out_labels + ['val_' + n for n in out_labels]
        print(' Callback metrics monitored by progbar')
        pp.pprint(callback_metrics)
        
        progbar = keras.callbacks.ProgbarLogger(count_mode='steps')
        progbar.set_model(self.keras_model)
        progbar.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': 1,
            'do_validation': False,
            'metrics': callback_metrics,
        })
        
        progbar.set_model(self.keras_model) 

        ## setup Checkpoint callback
        chkpoint = keras.callbacks.ModelCheckpoint(self.checkpoint_path, 
                                                   monitor='val_loss', verbose=1, save_best_only = True, save_weights_only=True)
        chkpoint.set_model(self.keras_model)


        
        progbar.on_train_begin()
        epoch_idx = self.epoch

        if epoch_idx >= epochs:
            print('Final epoch {} has already completed - Training will not proceed'.format(epochs))
        else:
            while epoch_idx < epochs :
                progbar.on_epoch_begin(epoch_idx)
                
                for steps_index in range(steps_per_epoch):
                    batch_logs = {}
                    # print(' self.epoch {}   epochs {}  step {} '.format(self.epoch, epochs, steps_index))
                    batch_logs['batch'] = steps_index
                    batch_logs['size']  = batch_size
                    progbar.on_batch_begin(steps_index, batch_logs)

                    train_batch_x, train_batch_y = next(train_generator)
                    
                    
                    outs = self.keras_model.train_on_batch(train_batch_x, train_batch_y)
    
                    if not isinstance(outs, list):
                        outs = [outs]
                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o
    
                    progbar.on_batch_end(steps_index, batch_logs)
                    
                    # print(outs)
                progbar.on_epoch_end(epoch_idx, {})
                # if (epoch_idx % 10) == 0:
                chkpoint.on_epoch_end(epoch_idx  , batch_logs)
                epoch_idx += 1

            # if epoch_idx != self.epoch:
            # chkpoint.on_epoch_end(epoch_idx -1, batch_logs)
            self.epoch = max(epoch_idx - 1, epochs)

            print('Final : self.epoch {}   epochs {}'.format(self.epoch, epochs))
        # end if (else)

        
        
    def compile(self, learning_rate, momentum, losses):
        '''
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        '''
        assert isinstance(losses, list) , "A loss function must be defined as the objective"
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=5.0)
        # optimizer= tf.train.GradientDescentOptimizer(learning_rate, momentum)

        ##------------------------------------------------------------------------
        ## Add Losses
        ## These are the losses aimed for minimization
        ##------------------------------------------------------------------------    
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}

        # loss_names = [  "rpn_class_loss", "rpn_bbox_loss"
                      # , "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"
                     # ]
        # loss_names = [ "fcn_loss", "fcn_norm_loss" ]
        
        loss_names = losses
                      
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            print('   keras model add loss for ', layer.output)
            self.keras_model.add_loss(tf.reduce_mean(layer.output, keepdims=True))

        ## Add L2 Regularization as loss to list of losses
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      for w in self.keras_model.trainable_weights
                      if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))


        ##------------------------------------------------------------------------    
        ## Compile
        ##------------------------------------------------------------------------    
        self.keras_model.compile(optimizer=optimizer, 
                                 loss=[None] * len(self.keras_model.outputs))

        ## Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(layer.output, keepdims=True))
        
        return


    def compile_only(self, learning_rate, layers):
        '''
        Compile the model without adding loss info
        learning_rate:  The learning rate to train with
        
        layers:         Allows selecting wich layers to train. It can be:
                        - A regular expression to match layer names to train
                        - One of these predefined values:
                        heads: The RPN, classifier and mask heads of the network
                        all: All the layers
                        3+: Train Resnet stage 3 and up
                        4+: Train Resnet stage 4 and up
                        5+: Train Resnet stage 5 and up
        '''
        # Use Pre-defined layer regular expressions
        if layers in self.layer_regex.keys():
            layers = self.layer_regex[layers]
            
        # Train
        log("Compile with learing rate; {} Learning Moementum: {} ".format(learning_rate,self.config.LEARNING_MOMENTUM))
        log("Checkpoint Folder:  {} ".format(self.checkpoint_path))
        
        self.set_trainable(layers)            
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)        

        out_labels = self.get_deduped_metrics_names()
        callback_metrics = out_labels + ['val_' + n for n in out_labels]
        print('Callback_metrics are:  ( val + _get_deduped_metrics_names() )\n')
        pp.pprint(callback_metrics)
        return
        
        
    def get_deduped_metrics_names(self):
        out_labels = self.keras_model.metrics_names

        # Rename duplicated metrics name
        # (can happen with an output layer shared among multiple dataflows).
        deduped_out_labels = []
        for i, label in enumerate(out_labels):
            new_label = label
            if out_labels.count(label) > 1:
                dup_idx = out_labels[:i].count(label)
                new_label += '_' + str(dup_idx + 1)
            deduped_out_labels.append(new_label)
        return deduped_out_labels        


        
