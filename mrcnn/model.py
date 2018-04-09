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
# import tensorflow.train as TT

import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
sys.path.append('..')

import mrcnn.utils            as utils
import mrcnn.loss             as loss
from   mrcnn.clsloss_layer    import CLSLossLayer
from   mrcnn.datagen          import data_generator
from   mrcnn.utils            import log
from   mrcnn.utils            import parse_image_meta_graph, parse_image_meta
from   mrcnn.FCN_model        import build_fcn_model, fcn_layer
from   mrcnn.RPN_model        import build_rpn_model
from   mrcnn.resnet_model     import resnet_graph
from   mrcnn.pcn_layer        import PCNLayer, PCILayer, PCNLayerTF
from   mrcnn.proposal_layer   import ProposalLayer
from   mrcnn.detect_tgt_layer import DetectionTargetLayer  
from   mrcnn.detect_layer     import DetectionLayer  
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
        self.mode      = mode
        self.config    = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

        print('>>> MaskRCNN initialization complete')

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        #------------------------------------------------------------------                            
        ## Build Inputs
        #------------------------------------------------------------------
        # input_image:
        # input_image_meta
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
            h, w = K.shape(input_image)[1], K.shape(input_image)[2]
            image_scale = K.cast(K.stack([h, w, h, w], axis=0), tf.float32)
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
        
        #----------------------------------------------------------------------------
        ## Resnet Backbone
        #----------------------------------------------------------------------------
        # Build the Resnet shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        #----------------------------------------------------------------------------
        Resnet_Layers      = resnet_graph(input_image, "resnet50", stage5=True)
        
        #----------------------------------------------------------------------------
        ## FPN network - Build the Feature Pyramid Network (FPN) layers.
        #----------------------------------------------------------------------------

        P2, P3, P4, P5, P6 = fpn_graph(Resnet_Layers)
        
        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps   = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        #----------------------------------------------------------------------------        
        ## Generate Anchors Box Coordinates
        # shape.anchors will contain an array of anchor box coordinates (y1,x1,y2,x2)
        #----------------------------------------------------------------------------        
        
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                      config.RPN_ANCHOR_RATIOS,
                                                      config.BACKBONE_SHAPES,
                                                      config.BACKBONE_STRIDES,
                                                      config.RPN_ANCHOR_STRIDE)


        #----------------------------------------------------------------------------        
        ## RPN Model - 
        #  model which is applied on the feature maps produced by the resnet backbone
        #----------------------------------------------------------------------------                
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
        print('>>> RPN Outputs ',  type(outputs))
        for i in outputs:
            print('     ', i.name)
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        #----------------------------------------------------------------------------                
        ## Proposal Layer
        #----------------------------------------------------------------------------                
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

        rpn_rois = ProposalLayer(proposal_count=proposal_count,            # num proposals to generate
                                 nms_threshold=config.RPN_NMS_THRESHOLD,
                                 name="proposal_rois",
                                 anchors=self.anchors,
                                 config=config)([rpn_class, rpn_bbox])
                                 
        #----------------------------------------------------------------------------                
        ## Training Mode
        #----------------------------------------------------------------------------                
        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image came from.
            _, _, _, active_class_ids = KL.Lambda(lambda x:  parse_image_meta_graph(x), mask=[None, None, None, None])(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs - Normalize and use ROIs provided as an input.
                input_rois  = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4], name="input_roi", dtype=np.int32)
                target_rois = KL.Lambda(lambda x: K.cast(x, tf.float32) / image_scale[:4])(input_rois)
            else:
                target_rois = rpn_rois
            
            #------------------------------------------------------------------------
            ## Generate detection targets
            #    generated RPNs ----> Target ROIs
            #
            #    target_* returned from this layer are the 'processed' versions of gt_*  
            # 
            #    Subsamples proposals and generates target outputs for training
            #    Note that proposal class IDs, input_normalized_gt_boxes, and gt_masks are zero padded. 
            #    Equally, returned rois and targets are zero padded.
            #------------------------------------------------------------------------
            rois, target_class_ids, target_bbox, target_mask = \
                DetectionTargetLayer(config, name="proposal_targets") \
                                    ([target_rois, input_gt_class_ids, input_normlzd_gt_boxes, input_gt_masks])
                    
            #------------------------------------------------------------------------
            ## MRCNN Network Heads
            #    TODO: verify that this handles zero padded ROIs
            #------------------------------------------------------------------------
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_feature_maps, config.IMAGE_SHAPE, config.POOL_SIZE, config.NUM_CLASSES)

            mrcnn_mask = \
                fpn_mask_graph(rois, mrcnn_feature_maps, config.IMAGE_SHAPE, config.MASK_POOL_SIZE, config.NUM_CLASSES)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            #------------------------------------------------------------------------
            ## PCN Layer to generate contextual feature maps using outputs from MRCNN 
            #------------------------------------------------------------------------            
            # Once we are comfortable with the results we can remove additional outputs from here.....
            pcn_gaussian, gt_gaussian, pcn_tensor, pcn_cls_cnt, gt_tensor, gt_cls_cnt = \
                 PCNLayer(config, name = 'cntxt_layer' )\
                         ([mrcnn_class, mrcnn_bbox, output_rois, input_gt_class_ids, input_normlzd_gt_boxes])            
              
            pcn_tensor2, pcn_cls_cnt2, gt_tensor2 , gt_cls_cnt2 = \
                 PCNLayerTF(config, name = 'cntxt_layer_2' )\
                         ([mrcnn_class, mrcnn_bbox, output_rois, input_gt_class_ids, input_normlzd_gt_boxes])
            
            print(' shape of pcn_tnesor2, cls_cnt2 ', pcn_tensor2.shape, pcn_cls_cnt2.shape)                         
            print(' shape of gt_tnesor2, gt_cls_cnt2 ', gt_tensor2.shape, gt_cls_cnt2.shape)                         
            #------------------------------------------------------------------------
            ## FCN Network Head
            #------------------------------------------------------------------------
            print(' shape of pcn_gaussian is ', pcn_gaussian.shape)

            
            # FCN_model = build_fcn_model(self.config)           
            # fcn_output = FCN_model(pcn_gaussian)

            # fcn_input = KL.Lambda(lambda x: x * 1, name="fcn_output2")(pcn_gaussian)
            # fcn_output = fcn_graph(pcn_gaussian, config)
        
            #------------------------------------------------------------------------
            ## Loss layer definitions
            #------------------------------------------------------------------------
            rpn_class_loss = KL.Lambda(lambda x: loss.rpn_class_loss_graph(*x),        name="rpn_class_loss")\
                            ([input_rpn_match, rpn_class_logits])
            
            ## The following two losses are the same, the only difference is the method of 
            ## calculating the Smooth L1 function, and should produce the same loss 
            ## can remove the old one when we're happy 
            rpn_bbox_loss  = KL.Lambda(lambda x: loss.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")\
                            ([input_rpn_bbox , input_rpn_match, rpn_bbox])
            
            rpn_bbox_loss_old = KL.Lambda(lambda x: loss.rpn_bbox_loss_graph_old(config, *x), name="rpn_bbox_loss_old") \
                            ([input_rpn_bbox , input_rpn_match, rpn_bbox])
            
            
            class_loss     = KL.Lambda(lambda x: loss.mrcnn_class_loss_graph(*x),      name="mrcnn_class_loss")\
                            ([target_class_ids, mrcnn_class_logits, active_class_ids])

            # A layer style implmentation of class_loss . Should produce the same loss
            # class_loss_2   = CLSLossLayer(config, name='mrcnn_class_loss_2') \
                            # ([target_class_ids, mrcnn_class_logits, active_class_ids])
            
            bbox_loss      = KL.Lambda(lambda x: loss.mrcnn_bbox_loss_graph(*x),       name="mrcnn_bbox_loss") \
                            ([target_bbox, target_class_ids, mrcnn_bbox])
            
            mask_loss      = KL.Lambda(lambda x: loss.mrcnn_mask_loss_graph(*x),       name="mrcnn_mask_loss") \
                            ([target_mask, target_class_ids, mrcnn_mask])
 
            # Model Inputs 
            inputs = [input_image,     #  
                      input_image_meta,
                      input_rpn_match ,         # [batch_sz, N, 1:<pos,neg,nutral>)                  [ 1,4092, 1]
                      input_rpn_bbox  ,         # [batch_sz, RPN_TRAIN_ANCHORS_PER_IMAGE, 4]         [ 1, 256, 4]
                      input_gt_class_ids,       # [batch_sz, MAX_GT_INSTANCES] Integer class IDs         [1, 100]
                      input_gt_boxes,           # [batch_sz, MAX_GT_INSTANCES, 4]                     [1, 100, 4]
                      input_gt_masks            # [batch_sz, height, width, MAX_GT_INSTANCES].  [1,  56, 56, 100]
                     ]
                      
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)

            outputs = [output_rois,
                       target_class_ids  , target_bbox  , target_mask  ,
                       pcn_gaussian      , gt_gaussian  , pcn_tensor   , pcn_cls_cnt, gt_tensor , gt_cls_cnt,
                       pcn_tensor2       , pcn_cls_cnt2 , gt_tensor2   , gt_cls_cnt2,
                       rpn_class_logits  , rpn_rois     , rpn_class    , rpn_bbox   ,
                       mrcnn_class_logits, mrcnn_class  , mrcnn_bbox   , mrcnn_mask ,
                       # fcn_output        ,
                       
                       rpn_class_loss    , rpn_bbox_loss     , rpn_bbox_loss_old,
                       class_loss , bbox_loss  , mask_loss
                       # , class_loss_2
                       ]
        
            # model = KM.Model(inputs, outputs, name='mask_rcnn')
        
        # end if Training

        
        #----------------------------------------------------------------------------                
        ## Inference Mode
        #----------------------------------------------------------------------------                
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, config.IMAGE_SHAPE,
                                     config.POOL_SIZE, config.NUM_CLASSES)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])
            # Convert boxes to normalized coordinates
            # TODO: let DetectionLayer return normalized coordinates to avoid
            #       unnecessary conversions
            h, w = config.IMAGE_SHAPE[:2]
            detection_boxes = KL.Lambda(lambda x: x[..., :4] / np.array([h, w, h, w]))(detections)

            # Create masks for detections
            mrcnn_mask = fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                        config.IMAGE_SHAPE,
                                        config.MASK_POOL_SIZE,
                                        config.NUM_CLASSES)

            # The PCI Layer prepares the input for the FCN layer ..
            pcn_gaussian, pcn_tensor, pcn_cls_cnt = \
                 PCILayer(config, name = 'cntxt_layer' ) ([mrcnn_class, mrcnn_bbox, detections])
                                        
            inputs  = [ input_image, input_image_meta]
            outputs = [ detections,
                        pcn_gaussian,
                        rpn_rois, rpn_class, rpn_bbox,
                        mrcnn_class, mrcnn_bbox, mrcnn_mask ]
            # model = KM.Model( inputs, outputs,  name='mask_rcnn')                           

            # end if Inference Mode
        
        model = KM.Model( inputs, outputs,  name='mask_rcnn')
        
        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        print('>>> MaskRCNN build complete')
        return model

##-------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------        
##-------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------        
        
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
        print('>>> find_last checkpoint file() ')
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
        log("    find_last info:   dir_name: {}".format(dir_name))
        log("    find_last info: checkpoint: {}".format(checkpoint))
        return dir_name, checkpoint

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

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()
        log('    load_weights: Log directory set to : {}'.format(filepath))
        # Update the log directory
        self.set_log_dir(filepath)
        print('>>> Load weights complete')        


    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        print('>>> Set_log_dir() -- model dir is ', self.model_dir)
        self.tb_dir = os.path.join(self.model_dir,'tensorboard')
        self.epoch  = 0
        now = datetime.datetime.now()
        
        # If we have a model path with date and epochs use them
        
        if model_path:
            # Continue from we left off. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            model_path = model_path.replace('\\' , "/")
            print('    set_log_dir: model_path (input) is : {}  '.format(model_path))        

            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1
            print('    set_log_dir: self.epoch set to {}  (Next epoch to run)'.format(self.epoch))
            print('    set_log_dir: tensorboard path: {}'.format(self.tb_dir))
            # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
                self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")
        log('    set_log_dir: Checkpoint path set to : {}'.format(self.checkpoint_path))

        
        
    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
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
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = utils.mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = utils.compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

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

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)
        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)
        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
        # Run object detection
        detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
            rois, rpn_class, rpn_bbox =\
            self.keras_model.predict([molded_images, image_metas], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

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
        return layers

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
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Run inference
        molded_images, image_metas, windows = self.mold_inputs(images)
        # TODO: support training mode?
        # if TEST_MODE == "training":
        #     model_in = [molded_images, image_metas,
        #                 target_rpn_match, target_rpn_bbox,
        #                 input_normalized_gt_boxes, gt_masks]
        #     if not config.USE_RPN_ROIS:
        #         model_in.append(target_rois)
        #     if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        #         model_in.append(1.)
        #     outputs_np = kf(model_in)
        # else:

        model_in = [molded_images, image_metas]
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        
        """
        Sets model layers as trainable if their names match the given
        regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("\nSelecting layers to train")
            log("{:5}    {:20}     {}".format( 'Layer', 'Layer Name', 'Layer Type'))

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for ind,layer in enumerate(layers):
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                # log("     [{}{:3}  {:20}   ({:20})  --- no weights, not trainable ]". \
                   # format(" " * indent, ind, layer.name,layer.__class__.__name__))

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
                    log("{}{:3}  {:20}   ({})".format(" " * indent, ind, layer.name,
                                            layer.__class__.__name__))
                else:
                    # log("     [{}{:3}  {:20}   ({:20})  --- not a layer we want to train ]". \
                        # format(" " * indent, ind, layer.name,layer.__class__.__name__))                
                    pass
                                            
    def train(self, 
              train_dataset, val_dataset, 
              learning_rate, 
              layers,
              epochs            = 0,
              epochs_to_run     = 0,
              batch_size        = 0, 
              steps_per_epoch   = 0):
        """Train the model.
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
        """
        assert self.mode == "training", "Create model in training mode."
        
        if batch_size == 0 :
            batch_size = self.config.BATCH_SIZE
        if epochs_to_run > 0 and epochs == 0:
            epochs = self.epoch + epochs_to_run
            
        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]
        if steps_per_epoch == 0:
            steps_per_epoch = self.config.STEPS_PER_EPOCH
            
        
        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=batch_size)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                        batch_size=batch_size,
                                        augment=False)

        my_callback = MyCallback()
        # Callbacks
        callbacks = [
              keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False)
            , keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            monitor='loss', verbose=1, save_best_only = True, save_weights_only=True)
            , my_callback
        ]

        # Train
        log("Starting at epoch {} of {} epochs. LR={}\n".format(self.epoch, epochs, learning_rate))
        log("Steps per epochs {} ".format(self.config.STEPS_PER_EPOCH))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

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
              epochs_to_run = 1, 
              batch_size = 0, 
              steps_per_epoch = 0):
        """Train the model.
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
        """
        assert self.mode == "training", "Create model in training mode."
        
       
        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]
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
       
        # Train
        log("Last epoch completed : {} ".format(self.epoch))
        log("Starting from epoch {} for {} epochs. LR={}".format(self.epoch, epochs_to_run, learning_rate))
        log("Steps per epoch:    {} ".format(steps_per_epoch))
        log("Batchsize      :    {} ".format(batch_size))
        log("Checkpoint Folder:  {} ".format(self.checkpoint_path))
        epochs = self.epoch + epochs_to_run
        
        from tensorflow.python.platform import gfile
        if not gfile.IsDirectory(self.log_dir):
            log('Creating checkpoint folder')
            gfile.MakeDirs(self.log_dir)
        else:
            log('Checkpoint folder already exists')
        
        self.set_trainable(layers)            
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)        
        
        out_labels = self.keras_model._get_deduped_metrics_names()
        callback_metrics = out_labels + ['val_' + n for n in out_labels]
        
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
        
        chkpoint = keras.callbacks.ModelCheckpoint(self.checkpoint_path, 
                                                   monitor='loss', verbose=1, save_best_only = True, save_weights_only=True)
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

    def compile(self, learning_rate, momentum):
        """
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=5.0)
        # optimizer= tf.train.GradientDescentOptimizer(learning_rate, momentum)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss",
                      "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            self.keras_model.add_loss(
                tf.reduce_mean(layer.output, keep_dims=True))

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      for w in self.keras_model.trainable_weights
                      if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, 
                                 loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(
                layer.output, keep_dims=True))



    def compile_only(self, 
              learning_rate, 
              layers):
        """Compile the model.
        learning_rate:  The learning rate to train with
        
        layers:         Allows selecting wich layers to train. It can be:
                        - A regular expression to match layer names to train
                        - One of these predefined values:
                        heads: The RPN, classifier and mask heads of the network
                        all: All the layers
                        3+: Train Resnet stage 3 and up
                        4+: Train Resnet stage 4 and up
                        5+: Train Resnet stage 5 and up
        """
        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]
            
        # Train
        log("Compile with learing rate; {} Learning Moementum: {} ".format(learning_rate,self.config.LEARNING_MOMENTUM))
        log("Checkpoint Folder:  {} ".format(self.checkpoint_path))
        
        self.set_trainable(layers)            
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)        
        # out_labels = self.keras_model._get_deduped_metrics_names()
        # callback_metrics = out_labels + ['val_' + n for n in out_labels]
        # print('Callback_metrics are:  ( val + _get_deduped_metrics_names() )\n')
        # pp.pprint(callback_metrics)
        return