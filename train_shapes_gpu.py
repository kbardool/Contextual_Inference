
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# ### Notes from implementation
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 
# 

# In[2]:


# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:90% !important; }</style>"))
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import random
import math
import re
import  gc
import time
import scipy.misc
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pprint
import argparse
import keras.backend as KB

sys.path.append('../')

import mrcnn.model     as modellib
import mrcnn.visualize as visualize
import mrcnn.shapes    as shapes

from mrcnn.config      import Config
from mrcnn.dataset     import Dataset 
from mrcnn.utils       import log, stack_tensors, stack_tensors_3d
from mrcnn.datagen     import data_generator, load_image_gt
from mrcnn.callbacks   import get_layer_output_1,get_layer_output_2
from mrcnn.visualize   import plot_gaussian
# from mrcnn.pc_layer    import PCTensor
# from mrcnn.pc_layer   import PCNLayer
print("Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))
pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4)

##------------------------------------------------------------------------------------
## process input arguments
##------------------------------------------------------------------------------------
# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')
# parser.add_argument("command",
                    # metavar="<command>",
                    # help="'train' or 'evaluate' on MS COCO")
# parser.add_argument('--dataset', required=True,
                    # metavar="/path/to/coco/",
                    # help='Directory of the MS-COCO dataset')
parser.add_argument('--model', required=False,
                    default='last',
                    metavar="/path/to/weights.h5",
                    help="'coco' , 'init' , or Path to weights .h5 file ")
# parser.add_argument('--logs', required=False,
                    # default=DEFAULT_LOGS_DIR,
                    # metavar="/path/to/logs/",
                    # help='Logs and checkpoints directory (default=logs/)')
# parser.add_argument('--limit', required=False,
                    # default=500,
                    # metavar="<image count>",
                    # help='Images to use for evaluation (defaults=500)')

parser.add_argument('--lr', required=False,
                    default=0.001,
                    metavar="<learning rate>",
                    help='Learning Rate (default=0.001)')

parser.add_argument('--epochs', required=False,
                    default=3,
                    metavar="<epochs to run>",
                    help='Number of epochs to run (default=3)')
                    
parser.add_argument('--steps_in_epoch', required=False,
                    default=1,
                    metavar="<steps in each epoch>",
                    help='Number of batches to run in each epochs (default=5)')
                    
args = parser.parse_args()
# args = parser.parse_args("train --dataset E:\MLDatasets\coco2014 --model mask_rcnn_coco.h5 --limit 10".split())
pp.pprint(args)
print("Model              :   ", args.model)
# print("Dataset: ", args.dataset)
# print("Logs:    ", args.logs)
# print("Limit:   ", args.limit)
print("Epochs to run      :   ", args.epochs)
print("Steps in each epoch:   ", args.steps_in_epoch)


##------------------------------------------------------------------------------------
## setup project directories
##------------------------------------------------------------------------------------
# # Root directory of the project ------------------------------------------------
ROOT_DIR = os.getcwd()
MODEL_PATH = 'E:\Models'
# Directory to save logs and trained model
MODEL_DIR = os.path.join(MODEL_PATH, "mrcnn_logs")
# Path to COCO trained weights
COCO_MODEL_PATH   = os.path.join(MODEL_PATH, "mask_rcnn_coco.h5")
RESNET_MODEL_PATH = os.path.join(MODEL_PATH, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

# LINUX MACHINE ------------------------------------------------------------------
# ROOT_DIR          = os.getcwd()
# MODEL_PATH        = os.path.expanduser('~/models')
# Directory to save logs and trained model
# MODEL_DIR         = os.path.join(MODEL_PATH, "mrcnn_logs")
# Path to COCO trained weights
# COCO_MODEL_PATH   = os.path.join(MODEL_PATH, "mask_rcnn_coco.h5")
# RESNET_MODEL_PATH = os.path.join(MODEL_PATH, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")


##------------------------------------------------------------------------------------
## setup tf session and debugging 
##------------------------------------------------------------------------------------
   # keras_backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))

# if 'tensorflow' == KB.backend():
    # from tensorflow.python import debug as tf_debug

    # config = tf.ConfigProto(
            # device_count = {'GPU': 0}
        # )
    # tf_sess = tf.Session(config=config)    
    # tf_sess = tf_debug.LocalCLIDebugWrapperSession(tf_sess)
    # KB.set_session(tf_sess)


#   tfconfig = tf.ConfigProto(
#               gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
#               device_count = {'GPU': 1}
#              )    
#     tfconfig = tf.ConfigProto()
#     tfconfig.gpu_options.allow_growth=True
#     tfconfig.gpu_options.visible_device_list = "0"
#     tfconfig.gpu_options.per_process_gpu_memory_fraction=0.5
#     tf_sess = tf.Session(config=tfconfig)
#     set_session(tf_sess)
##------------------------------------------------------------------------------------


##------------------------------------------------------------------------------------
## Build configuration object 
##------------------------------------------------------------------------------------
config = shapes.ShapesConfig()
config.BATCH_SIZE      = 2                  # Batch size is 2 (# GPUs * images/GPU).
config.IMAGES_PER_GPU  = 2                  # Must match BATCH_SIZE
config.STEPS_PER_EPOCH = int(args.steps_in_epoch)
config.EPOCHS_TO_RUN   = int(args.epochs)
config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]
config.LEARNING_RATE   = float(args.lr)

##------------------------------------------------------------------------------------
## Build shape dataset        
##------------------------------------------------------------------------------------
# Training dataset
# generate 500 shapes 
dataset_train = shapes.ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = shapes.ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

##------------------------------------------------------------------------------------    
## Load and display random samples
##------------------------------------------------------------------------------------
# image_ids = np.random.choice(dataset_train.image_ids, 3)
# for image_id in [3]:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

##------------------------------------------------------------------------------------
## Build Model
##------------------------------------------------------------------------------------

try :
    del model
    gc.collect()
except: 
    pass
KB.clear_session()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

#model.keras_model.summary(line_length = 120) 

##------------------------------------------------------------------------------------
## Load Model hf5 file 
##------------------------------------------------------------------------------------
# KB.set_learning_phase(1)

# Which weights to start with?
init_with = args.model  # imagenet, coco, or last

if init_with == "imagenet":
#     loc=model.load_weights(model.get_imagenet_weights(), by_name=True)
    loc=model.load_weights(RESNET_MODEL_PATH, by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes

    # See README for instructions to download the COCO weights
    loc=model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
# Load the last model you trained and continue training                               
elif init_with == "last":
    lastChkPointFile = model.find_last()[1]
    print('    Last Checkpoint file output: ', lastChkPointFile)
    loc= model.load_weights(lastChkPointFile, by_name=True)
print('    Load weights complete :', loc)
print('    Load weights complete :', model.set_log_dir )


print()
print("    Model                 : ", args.model)
# print("Dataset: ", args.dataset)
# print("Logs:    ", args.logs)
# print("Limit:   ", args.limit)
print("    Epochs to run         : ", config.EPOCHS_TO_RUN)
print("    Steps in each epoch   : ", config.STEPS_PER_EPOCH)
print()
print('    Root dir              : ', ROOT_DIR)
print('    Model path            : ', MODEL_PATH)
print('    Model dir             : ', MODEL_DIR)
print('    COCO Model Path       : ', COCO_MODEL_PATH)
print('    Resnet Model Path     : ', RESNET_MODEL_PATH)
print('    Checkpoint folder Path: ', MODEL_DIR)

config.display() 


##------------------------------------------------------------------------------------
## Training heads using fit_generator()
##------------------------------------------------------------------------------------

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

# model.train(dataset_train, dataset_val, 
            # learning_rate=config.LEARNING_RATE, 
            # epochs = 10,
            #  # epochs_to_run =2, 
            # layers='heads')


##------------------------------------------------------------------------------------
##  Training heads using train_in_batches ()
##------------------------------------------------------------------------------------
# 
# We need to use this method for the time being as the fit generator does not have 
# provide EASY access to the output in Keras call backs. By training in batches, we pass
# a batch through the network, pick up the generated RoI detections and bounding boxes 
# and generate our semantic / gaussian tensors ...
# 
model.train_in_batches(dataset_train, dataset_val, 
                       learning_rate = config.LEARNING_RATE, 
                       epochs_to_run = config.EPOCHS_TO_RUN,
                       layers='heads')


# ## Simulate one training iteration - 1

# In[6]:

'''
from mrcnn.datagen import data_generator, load_image_gt
np.set_printoptions(linewidth=100)

learning_rate   = model.config.LEARNING_RATE
epochs_to_run   = 2
layers          = 'heads'
batch_size      = 0
steps_per_epoch = 0

# assert self.mode == "training", "Create model in training mode."
# Pre-defined layer regular expressions
layer_regex = {
    # all layers but the backbone
    "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(fcn\_.*)",
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
    batch_size = model.config.BATCH_SIZE            
if steps_per_epoch == 0:
    steps_per_epoch = model.config.STEPS_PER_EPOCH

# Data generators
train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                 batch_size=batch_size)
val_generator   = data_generator(dataset_val, model.config, shuffle=True,
                                 batch_size=batch_size,
                                 augment=False)

# Train
log("Last epoch completed : {} ".format(model.epoch))
log("Starting from epoch {} for {} epochs. LR={}".format(model.epoch, epochs_to_run, learning_rate))
log("Steps per epoch:    {} ".format(steps_per_epoch))
log("Batchsize      :    {} ".format(batch_size))
log("Checkpoint Folder:  {} ".format(model.checkpoint_path))
epochs = model.epoch + epochs_to_run

from tensorflow.python.platform import gfile
if not gfile.IsDirectory(model.log_dir):
    log('Creating checkpoint folder')
    gfile.MakeDirs(model.log_dir)
else:
    log('Checkpoint folder already exists')

model.set_trainable(layers)            
model.compile(learning_rate, model.config.LEARNING_MOMENTUM)        

out_labels = model.get_deduped_metrics_names()
callback_metrics = out_labels + ['val_' + n for n in out_labels]

progbar = keras.callbacks.ProgbarLogger(count_mode='steps')
progbar.set_model(model.keras_model)
progbar.set_params({
    'epochs': epochs,
    'steps': steps_per_epoch,
    'verbose': 1,
    'do_validation': False,
    'metrics': callback_metrics,
})

progbar.set_model(model.keras_model) 

chkpoint = keras.callbacks.ModelCheckpoint(model.checkpoint_path, 
                                           monitor='loss', verbose=1, save_best_only = True, save_weights_only=True)
chkpoint.set_model(model.keras_model)

progbar.on_train_begin()
epoch_idx = model.epoch

'''

# ## Simulate one training iteration - 2

# In[7]:

'''
if epoch_idx >= epochs:
    print('Final epoch {} has already completed - Training will not proceed'.format(epochs))

# while epoch_idx < epochs :
progbar.on_epoch_begin(epoch_idx)
steps_index = 0
# for steps_index in range(steps_per_epoch):

batch_logs = {}
print(' self.epoch {}   epochs {}  step {} '.format(model.epoch, epochs, steps_index))
batch_logs['batch'] = steps_index
batch_logs['size']  = batch_size
progbar.on_batch_begin(steps_index, batch_logs)

'''

# In[8]:




# ## Simulate one training iteration - 3

# In[ ]:
'''
train_batch_x, train_batch_y = next(train_generator)


imgmeta_idx= model.keras_model.input_names.index('input_image_meta')
img_meta  =  train_batch_x[imgmeta_idx]

image_id = img_meta[0,0]
print('Image id: ',image_id)
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)
visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

image_id = img_meta[1,0]
print('Image id: ',image_id)
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)
visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

outs = model.keras_model.train_on_batch(train_batch_x, train_batch_y)
'''

# ### Simulate one training iteration - 4

# In[ ]:

'''
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
model.epoch = max(epoch_idx - 1, epochs)

print('Final : self.epoch {}   epochs {}'.format(model.epoch, epochs))
# end if (else)
'''

# In[ ]:

'''
epoch_idx = 0


# ### Define Data Generator

# In[ ]:


train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                 batch_size=model.config.BATCH_SIZE,
                                 augment = False)
val_generator = data_generator(dataset_val, model.config, shuffle=True, 
                                batch_size=model.config.BATCH_SIZE,
                                augment=False)


# ### Get next shapes from generator and display loaded shapes

# In[ ]:


train_batch_x, train_batch_y = next(train_generator)


# In[ ]:


# train_batch_x, train_batch_y = next(train_generator)
imgmeta_idx = model.keras_model.input_names.index('input_image_meta')
img_meta    = train_batch_x[imgmeta_idx]

for img_idx in range(config.BATCH_SIZE):
    image_id = img_meta[img_idx,0]
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    print('Image id: ',image_id)
    print('Image meta', img_meta[img_idx])
    print('Classes (1: circle, 2: square, 3: triangle ): ',class_ids)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# ### Push Data thru model using get_layer_output()

# In[ ]:



layers_out = get_layer_output_2(model.keras_model, train_batch_x, 1)


# 
# ## Plot Predicted and Ground Truth Probability Heatmaps `pred_gaussian` and `gt_gaussian` (Tensorflow)
# 
# `pred_gaussian2` and `gt_gaussian2` from Tensorflow PCN layer

# In[ ]:


# gt_heatmap  = layers_out[27]     # gt_gaussiam 
# pred_heatmap= layers_out[24]  # pred_gaussian
gt_heatmap  = layers_out[21]     # gt_gaussiam 
pred_heatmap= layers_out[18]  # pred_gaussian
print('gt_gaussian heatmap shape : ', gt_heatmap.shape, ' pred_gaussian heatmap shape: ', pred_heatmap.shape)
num_images = 1 # config.IMAGES_PER_GPU
num_classes = config.NUM_CLASSES

img = 2

image_id = img_meta[img,0]
print('Image id: ',image_id)
print('Classes (1: circle, 2: square, 3: triangle ): ')
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)
visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


for cls in range(num_classes):
    ttl = 'GROUND TRUTH HEATMAP - image :  {} class: {} '.format(img,cls)
    print(' *** Zout  ', gt_heatmap[img,:,:,cls].shape, ttl)   
    plot_gaussian( gt_heatmap[img,:,:,cls], title = ttl)
    
    ttl = 'PREDICTED heatmap  - image :  {} class: {} '.format(img,cls)     
    print(' *** pred_heatmap ', pred_heatmap[img,:,:,cls].shape, ttl)   
    plot_gaussian(pred_heatmap[img,:,:,cls], title = ttl)  


# 
# ### Plot Output from FCN network `fcn_bilinear` and compare with `pred_gaussian`

# In[ ]:


from mrcnn.visualize import plot_gaussian
import matplotlib as plt

get_ipython().run_line_magic('matplotlib', 'inline')
img = 2
image_id = img_meta[img,0]
print('Image id: ',image_id)
print('Classes (1: circle, 2: square, 3: triangle ): ')
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)
visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


Zout  = layers_out[21]     # gt_gaussiam 
Zout2 = layers_out[12]     # fcn_bilinear

print(Zout.shape, Zout2.shape)
num_images = config.IMAGES_PER_GPU
num_classes = config.NUM_CLASSES


for cls in range(num_classes):
    ttl = 'GroundTruth - image :  {} class: {} '.format(img,cls)
    print(' *** Zout  ', Zout[img,:,:,cls].shape, ttl)   
    plot_gaussian( Zout[img,:,:,cls], title = ttl)
    
    ttl = 'FCN_Bilinear- image :  {} class: {} '.format(img,cls)     
    print(' *** Zout2 ', Zout2[img,:,:,cls].shape, ttl)   
    plot_gaussian(Zout2[img,:,:,cls], title = ttl)  


# ### Display ground truth bboxes from Shapes database (using `load_image_gt` )
# 
# Here we are displaying the ground truth bounding boxes as provided by the dataset

# In[ ]:


img = 0
image_id = img_meta[img,0]
print('Image id: ',image_id)
p_original_image, p_image_meta, p_gt_class_id, p_gt_bbox, p_gt_mask =              load_image_gt(dataset_train, config, image_id, augment=False, use_mini_mask=True)
# print(p_gt_class_id.shape, p_gt_bbox.shape, p_gt_mask.shape)
print(p_gt_bbox[0:3,:])
print(p_gt_class_id)
visualize.draw_boxes(p_original_image, p_gt_bbox[0:3])

# image_id = img_meta[img,0]
# print('Image id: ',image_id)
# p_original_image, p_image_meta, p_gt_class_id, p_gt_bbox, p_gt_mask =  \
#             load_image_gt(dataset_train, config, image_id, augment=False, use_mini_mask=True)
# # print(p_gt_class_id.shape, p_gt_bbox.shape, p_gt_mask.shape)
# print(p_gt_bbox)
# print(p_gt_class_id)
# visualize.draw_boxes(p_original_image, p_gt_bbox)


# ### Display Predicted  Ground Truth Bounding Boxes  `gt_tensor` and `gt_tensor2`
# 
# layers_out[22]  `gt_tensor` is based on input_gt_class_ids and input_normlzd_gt_boxes
# layers_out[28]  `gt_tensor2` is based on input_gt_class_ids and input_normlzd_gt_boxes, generated using Tensorflow
# 
# Display the Ground Truth bounding boxes from the tensor we've constructed

# In[ ]:


from mrcnn.utils  import stack_tensors, stack_tensors_3d
# print(gt_bboxes)
# visualize.display_instances(p_original_image, p_gt_bbox, p_gt_mask, p_gt_class_id, 
#                             dataset_train.class_names, figsize=(8, 8))
# pp.pprint(gt_bboxes)
img = 0
image_id = img_meta[img,0]

print('Image id: ',image_id)
p_image, p_image_meta, p_gt_class_id, p_gt_bbox, p_gt_mask =              load_image_gt(dataset_train, config, image_id, augment=False, use_mini_mask=True)   
gt_bboxes_stacked = stack_tensors_3d(layers_out[22][img])
print(gt_bboxes_stacked)
visualize.draw_boxes(p_image, gt_bboxes_stacked[0:2,2:6])




# In[ ]:


layers_out[0][0] * [128, 128,128,128]   #output_rois*


# ### Calculate  mrcnn_bbox_loss

# In[ ]:


import keras.backend as K

from mrcnn.utils import apply_box_deltas
from mrcnn.loss  import smooth_l1_loss

target_class_ids = layers_out[1][0:1]
target_bbox      = layers_out[2][0:1]
mrcnn_bbox       = layers_out[10][0:1]
mrcnn_class_ids  = np.argmax(layers_out[9][0:1],axis = -1)     # mrcnn_class_ids

print('target_class_ids', target_class_ids.shape)
print(target_class_ids)  # tgt_class_ids
print(' class with max probability', mrcnn_class_ids.shape)
print(mrcnn_class_ids)
print('target_bboxes', target_bbox.shape)
# print(target_bbox)  # tgt_bounding boxes
print('mrcnn_bboxes',mrcnn_bbox.shape)
# print(mrcnn_bbox)  #mrcnn_bboxes
pred_bbox = mrcnn_bbox

# calc mrcnn_bbox_loss
target_class_ids = K.reshape(target_class_ids, (-1,))
print(target_class_ids.shape)
target_bbox      = K.reshape(target_bbox, (-1, 4))
print('target_bboxx: ', target_bbox.shape)
pred_bbox        = K.reshape(pred_bbox, (-1, pred_bbox.shape[2], 4))
print('pred_bbox : ', pred_bbox.shape)

positive_roi_ix        = tf.where(target_class_ids > 0)[:, 0]
print(positive_roi_ix.eval())
positive_roi_class_ids = tf.cast( tf.gather(target_class_ids, positive_roi_ix), tf.int64)
print(positive_roi_class_ids.eval())
indices                = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)
print(indices.eval())


target_bbox = tf.gather(target_bbox, positive_roi_ix)
print(target_bbox.eval())
pred_bbox   = tf.gather_nd(pred_bbox, indices)
print(pred_bbox.eval())

print('tf.size ',tf.size(target_bbox).eval())

diff = K.abs(target_bbox - pred_bbox)
print(diff.eval())

less_than_one = K.cast(K.less(diff, 1.0), "float32")
# print(less_than_one.eval())

loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
# print( (1-less_than_one).eval())



# loss        = K.switch(tf.size(target_bbox) > 0,
#                 smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
#                 tf.constant(0.0))
print(loss.eval())
sumloss = K.sum(loss)
print(sumloss.eval())
print((sumloss/40).eval())
meanloss        = K.mean(loss)
print(meanloss.eval())


# ###  Calculate mrcnn_class_loss

# In[ ]:


import keras.backend as K

from mrcnn.utils import apply_box_deltas
from mrcnn.loss  import smooth_l1_loss

target_class_ids = layers_out[1][0:1]
pred_class_logits = layers_out[8][0:1]
active_class_ids    = np.array([1,1,1,1])

# mrcnn_class_ids  = np.argmax(layers_out[9][0:1],axis = -1)     # mrcnn_class_ids

print(' target_class_ids', target_class_ids.shape)
print(target_class_ids)  # tgt_class_ids
print(' class logits', pred_class_logits.shape)
print(pred_class_logits)
print(' active, class_ids ', active_class_ids.shape)
print(active_class_ids)  # tgt_bounding boxes

pred_class_ids = tf.argmax(pred_class_logits, axis=2)
print(pred_class_ids.eval())  #mrcnn_bboxes
mrcnn_class_ids  = np.argmax(layers_out[9][0:1],axis = -1)     # mrcnn_class_ids
print(mrcnn_class_ids)
# pred_bbox = mrcnn_bbox
pred_active = tf.to_float(tf.gather(active_class_ids, pred_class_ids))
print(pred_active.eval())
# calc mrcnn_bbox_loss
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
       labels=target_class_ids, logits=pred_class_logits)
print(loss.eval())

loss = loss * tf.to_float(pred_active)
print(loss.eval())

print(tf.reduce_sum(loss).eval())
print(tf.reduce_sum(pred_active).eval())
loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
print(loss.eval())


# ###  Calculate mrcnn_mask_loss

# In[ ]:


import keras.backend as K

from mrcnn.utils import apply_box_deltas
from mrcnn.loss  import smooth_l1_loss

target_class_ids    = layers_out[1][0:3]
target_masks        = layers_out[3][0:3]
pred_masks          = layers_out[11][0:3]
# mrcnn_class_ids  = np.argmax(layers_out[9][0:1],axis = -1)     # mrcnn_class_ids
print('    target_class_ids shape :', target_class_ids.shape)
print('    target_masks     shape :', target_masks.shape)
print('    pred_masks       shape :', pred_masks.shape)    


target_class_ids = K.reshape(target_class_ids, (-1,))
print('    target_class_ids shape :', target_class_ids.shape, '\n', target_class_ids.eval())

mask_shape       = tf.shape(target_masks)
print('    mask_shape       shape :', mask_shape.shape, mask_shape.eval())    

target_masks     = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
print('    target_masks     shape :', tf.shape(target_masks).eval())        

pred_shape       = tf.shape(pred_masks)
print('    pred_shape       shape :', pred_shape.shape, pred_shape.eval())        

pred_masks       = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
print('    pred_masks       shape :', tf.shape(pred_masks).eval())        


pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])
print('    pred_masks       shape :', tf.shape(pred_masks).eval())        

# Only positive ROIs contribute to the loss. And only
# the class specific mask of each ROI.
positive_ix        = tf.where(target_class_ids > 0)[:, 0]
positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
indices            = tf.stack([positive_ix, positive_class_ids], axis=1)
print(indices.eval())



y_true = tf.gather(target_masks, positive_ix)
print('     y_true shape:', tf.shape(y_true).eval())
y_pred = tf.gather_nd(pred_masks, indices)
print('     y_pred shape:', tf.shape(y_pred).eval())

loss = K.switch(tf.size(y_true) > 0,
                K.binary_crossentropy(target=y_true, output=y_pred),
                tf.constant(0.0))
print(tf.shape(loss).eval())

loss = K.mean(loss)
print('     final loss shape:', tf.shape(loss).eval())
print(loss.eval())
loss = K.reshape(loss, [1, 1])
print('     final loss shape:', tf.shape(loss).eval())
print(loss.eval())


# ### Calculate a pixel loss on fcn_gaussian and gt_gaussian 

# In[ ]:


import keras.backend as K

from mrcnn.utils import apply_box_deltas
from mrcnn.loss  import smooth_l1_loss
pred_masks          = layers_out[12][0:3]
target_masks        = layers_out[27][0:3]

print('    target_masks     shape :', tf.shape(target_masks).eval())
print('    pred_masks       shape :', tf.shape(pred_masks).eval())    

diff = K.abs(target_masks - pred_masks)
print(tf.shape(diff).eval())

less_than_one = K.cast(K.less(diff, 1.0), "float32")
print(tf.shape(less_than_one).eval())

loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
print(tf.shape(loss).eval())

# print( (1-less_than_one).eval())

# loss = K.switch(tf.size(y_true) > 0,
#                 K.binary_crossentropy(target=y_true, output=y_pred),
#                 tf.constant(0.0))
meanloss = K.mean(loss)
print(tf.shape(meanloss).eval())
print(meanloss.eval())
# loss = K.reshape(loss, [1, 1])
# print('     final loss shape:', loss.get_shape())
# return loss


mask_shape       = tf.shape(target_masks)
print('    mask_shape       shape :', tf.shape(mask_shape).eval())    

target_masks     = K.reshape(target_masks, (-1, mask_shape[1], mask_shape[2]))
print('    target_masks     shape :', tf.shape(target_masks).eval())        

pred_shape       = tf.shape(pred_masks)
print('    pred_shape       shape :', tf.shape(pred_shape).eval())        

pred_masks       = K.reshape(pred_masks, (-1, pred_shape[1], pred_shape[2]))
print('    pred_masks       shape :', tf.shape(pred_masks).eval())
# Permute predicted masks to [N, num_classes, height, width]
# diff = K.abs(target_masks - pred_masks)
# print(tf.shape(diff).eval())

# less_than_one = K.cast(K.less(diff, 1.0), "float32")
# print(tf.shape(less_than_one).eval())

# loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
# print(tf.shape(loss).eval())

# meanloss = K.mean(loss)
# print(tf.shape(meanloss).eval())
# print(meanloss.eval())

loss = K.switch(tf.size(target_masks) > 0,
                smooth_l1_loss(y_true=target_masks, y_pred=pred_masks),
                tf.constant(0.0))
loss = K.mean(loss)
loss = K.reshape(loss, [1, 1])
print('     final loss shape:', loss.get_shape())
print(loss.eval())


# ###  Mean values of GT, Pred, and FCN heatmaps 

# In[ ]:


pred_masks = tf.identity(layers_out[24])
gt_masks = tf.identity(layers_out[27])
fcn_masks = tf.identity(layers_out[12])
print(gt_masks.shape, fcn_masks.shape)
for img in range(5):
    for cls in range(4):
        gt_mean = K.mean(gt_masks[img,:,:,cls])
        fcn_mean= K.mean(fcn_masks[img,:,:,cls])
        pred_mean= K.mean(pred_masks[img,:,:,cls])
        print('Img/Cls: ', img, '/', cls,'    gtmean: ', gt_mean.eval(), '\t fcn : ' , fcn_mean.eval(), '\t pred :', pred_mean.eval())


# In[ ]:


img  = 0
class_probs = layers_out[9][img]   # mrcnn_class
deltas      = layers_out[10][img]       # mrcnn_bbox

print(class_probs.shape)
print('class probabilities')
print(class_probs)
class_ids = np.argmax(layers_out[9][img],axis = 1)     # mrcnn_class_ids
print(' class with max probability')
print(class_ids)


# layers_out[10][2,0,3]
print('deltas.shape :', deltas.shape)
print(deltas[0:4])

deltas_specific = deltas[np.arange(32),class_ids]
print('deltas of max prob class: ', deltas_specific.shape)
print(deltas_specific[0:5])
output_rois = layers_out[0][img]*[128,128,128,128]
print('output_rois: ', output_rois.shape)
print(output_rois[0:])

refined_rois    = apply_box_deltas(output_rois, deltas_specific * config.BBOX_STD_DEV)
print('refined rois: ',refined_rois.shape)
print(refined_rois)


# In[ ]:


layers_out[30][2]


# In[ ]:


img = 0
cls = 0
fcn_out = layers_out[12][img]
fcn_sum = np.sum(fcn_out, axis=(0,1))
print(fcn_sum)
for cls in range(4):
    print('min :', np.min(fcn_out[:,:,cls]), 'max :', np.max(fcn_out[:,:,cls]), )


# In[ ]:


print(train_batch_x[4][2])
print(train_batch_x[5][2]/[128,128,128,128])


# ## Fine Tuning
# Fine tune all layers

# In[ ]:


# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=211,
            layers="all")


# ## Save 

# In[ ]:


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
model.keras_model.save_weights(model_path)
'''
