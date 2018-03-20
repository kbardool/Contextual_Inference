import keras.backend as KB
import pprint

from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

pp = pprint.PrettyPrinter(indent=2, width=100)

def show_modelstuff( model ):
    # print(KB.eval(KB.learning_phase()))
    KB.set_learning_phase(1)
    print(' Learning phase value is: ' ,KB.learning_phase())
    # model = model.keras_model
    print('\n Metrics: ') 
    pp.pprint(model._get_deduped_metrics_names())
    print('\n Outputs: ') 
    pp.pprint(model.outputs)
    # pp.pprint(model.fit_generator.__dict__['__wrapped__'])
    print('\n Losses: ') 
    # pp.pprint(model.losses)
    pp.pprint(model.metrics_names)

# import numpy as np
# def get_layer_output(model, model_input,output_layer, training_flag = True):
    # _my_input = model_input 
    # for name,inp in zip(model.input_names, model_input):
        # print(' Input Name:  ({:24}) \t  Input shape: {}'.format(name, inp.shape))


    # _mrcnn_class = KB.function(model.input , model.output)
# #                               [model.keras_model.layers[output_layer].output])
    # output = _mrcnn_class(_my_input)                  
    
    # for name,out in zip (model.output_names,output):
        # print(' Output Name: ({:24}) \t Output shape: {}'.format(name, out.shape))
    # return output
    
def bbox_gaussian( bbox, Zin ):
    """
    receive a bounding box, and generate a gaussian distribution centered on the bounding box and with a 
    covariance matrix based on the width and height of the bounding box/. 
    Inputs : 
    --------
    bbox :  (index, class_id, class_prob, y1, x1, y2, x2)
    bbox :  (index, class_id, class_prob, cx, cy, width, height)
    Returns:
    --------
    bbox_g  grid mesh [image_height, image width] covering the distribution

    """
    print(bbox.shape)
    width  = bbox[6] - bbox[4]
    height = bbox[5] - bbox[3]
    cx     = bbox[4] + ( width  / 2.0)
    cy     = bbox[3] + ( height / 2.0)
#     cx, cy, width, height = bbox[3:]
    print('center is ({},{}) width: {}  height: {} '.format(cx, cy, width,  height))
#     srtd_cpb_2 = np.column_stack((srtd_cpb[:, 0:2], cx,cy, width, height ))
    X = np.arange(0, 128, 1)
    Y = np.arange(0, 128, 1)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,0] = X;
    pos[:,:,1] = Y;

    rv = multivariate_normal([cx,cy],[[12,0.0] , [0.0,19]])
    Zout  = rv.pdf(pos)
    Zout += Zin
    return Zout        



def plot_gaussian( Z ):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    X = np.arange(0, 128, 1)
    Y = np.arange(0, 128, 1)
    X, Y = np.meshgrid(X, Y)
    
    pos = np.empty(X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,0] = X;
    pos[:,:,1] = Y;
    surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    # # Customize the z axis.
    ax.set_zlim(0.0 , 0.05)
    ax.set_ylim(0,130)
    ax.set_xlim(0,130)
    ax.set_xlabel(' X axis')
    ax.set_ylabel(' Y axis')
    ax.invert_yaxis()
    ax.view_init(elev=140, azim=-88)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    
def stack_tensor(model):
    pred_cpb_all = np.empty((0,8))
    for i in range(1,model.config.NUM_CLASSES):
        if pred_cls_cnt[i] > 0:
            pred_cpb_all = np.vstack((pred_cpb_all, pred_cpb[i,0:pred_cls_cnt[i]] ))

from scipy.stats import  multivariate_normal


#----------------------------------------------------------------------------------------------
# Generate prediction tensor
#----------------------------------------------------------------------------------------------

# // pass model to TensorBuilder

# h, w = config.IMAGE_SHAPE[:2]
# pc_tensor = PCTensor(model)
# class_idx = mm.output_names.index('mrcnn_class')
# bbox_idx  = mm.output_names.index('mrcnn_bbox')
# outroi_idx= mm.output_names.index('output_rois')


# print('mrcnn_class idx: {}   mrcnn_bbox idx : {}   output_rois idx : {}'.format(class_idx, bbox_idx,outroi_idx))


# mrcnn_class = ppp[class_idx]
# mrcnn_bbox  = ppp[bbox_idx]
# rois_norm   = ppp[outroi_idx][0,...] 
# rois        = rois_norm * np.array([h,w,h,w])

# num_classes = config.NUM_CLASSES
# num_rois    = config.TRAIN_ROIS_PER_IMAGE
# num_max_gt  = config.DETECTION_MAX_INSTANCES
# num_cols    = 8 

# pred_arr    = np.zeros((num_classes, num_rois, num_cols ))      # 4, 32, 7
# pred_cpb    = np.zeros_like(pred_arr)
# pred_cls_cnt= np.zeros((num_classes), dtype='int16')

# print('mrcnn_bbox shape is : ',mrcnn_bbox.shape, ' pred_cpb shape is   : ',pred_cpb.shape  )

# # use the argmaxof each row to determine the dominating (predicted) class
# #---------------------------------------------------------------------------
# pred_class = np.argmax(mrcnn_class[0,:,:],axis=1).astype('int16')   # (32,)

# # pred_index = np.arange(pred_class.shape[0],dtype='int16')
# # pred_prob  =    np.max(mrcnn_class[0,:,:],axis=1)                   #  (32,)
# # dont need it for now. Need to see if and how we should apply  the delta to the bounding box coords
# # pred_delta   = mrcnn_bbox[0,pred_index[:],pred_class[:],:]        

# for i in range(num_classes) :
    # class_idxs = np.argwhere(pred_class == i )
    # pred_cls_cnt[i] = class_idxs.shape[0] 
    # for j , c_idx in enumerate(class_idxs):
        # pred_arr[i, j,  0]  = j
        # pred_arr[i, j,  1]  = i                                   # class_id
        # pred_arr[i, j,  2]  = np.max(mrcnn_class[0, c_idx ])      # probability
        # pred_arr[i, j,3:7]  = rois[c_idx]                         # roi coordinates
        # pred_arr[i, j,  7]  = c_idx                               # index from mrcnn_class array (temp for verification)
        
        
# # sort each class in descending prediction order 

# order = pred_arr[:,:,2].argsort()

# for i in range(num_classes):
    # pred_cpb[i,:,1:] =  pred_arr[i,order[i,::-1],1:]      
# pred_cpb[:,:,0] = pred_arr[:,:,0]

# print('pred_cpb shape', pred_cpb.shape)

# pred_cpb_all = np.empty((0,8))
# for i in range(1,num_classes):
    # if pred_cls_cnt[i] > 0:
        # pred_cpb_all = np.vstack((pred_cpb_all, pred_cpb[i,0:pred_cls_cnt[i]] ))
        
# #----------------------------------------------------------------------------------------------
# # display prediction tensor
# #----------------------------------------------------------------------------------------------        
# # Display values for sanity check 
# # i = 0
# print(pred_cls_cnt)
# print(' pred_cpb_all ')
# print(pred_cpb_all)
# print(' pred_cpb ')
# print(pred_cpb[i])        
        
        
#----------------------------------------------------------------------------------------------
# Generate ground truth tensor
#----------------------------------------------------------------------------------------------        
# gtcls_idx = mm.input_names.index('input_gt_class_ids')
# gtbox_idx = mm.input_names.index('input_gt_boxes')
# gtmsk_idx = mm.input_names.index('input_gt_masks')
# print('gtcls_idx: ',gtcls_idx, 'gtbox_idx :', gtbox_idx)
# gt_classes = sample_x[gtcls_idx][0,:]
# gt_bboxes  = sample_x[gtbox_idx][0,:,:]

# gt_cpb     = np.zeros((num_classes, num_max_gt, num_cols ))      # 4, 32, 7
# gt_cls_cnt = np.zeros((num_classes), dtype='int16')
# # gt_masks   = sample_x[gtmsk_idx][0,:,:,nz_idx]
# # gt_indexes = np.arange(gt_classes.shape[0],dtype='int16')
# # gt_probs   = np.ones(gt_classes.shape[0])

# print('gt_classes.shape :',gt_classes.shape, 'gt_boxes.shape :',gt_bboxes.shape,'gt_masks.shape :', gt_masks.shape)
 
# for i in range(num_classes) :
    # print('indexes for class',i )
    # class_idxs = np.argwhere(gt_classes == i )
    # gt_cls_cnt[i] = class_idxs.shape[0]
    # for j , c_idx in enumerate(class_idxs):
        # gt_cpb[i, j,  0]  = j
        # gt_cpb[i, j,  1]  = i                                   # class_id
        # gt_cpb[i, j,  2]  = 1.0                                 # probability
        # gt_cpb[i, j, 3:7] = gt_bboxes[c_idx,:]                         # roi coordinates
        # gt_cpb[i, j,  7]  = c_idx                               # index from mrcnn_class array (temp for verification)

# gt_cpb_all = np.empty((0,8))
# for i in range(1,num_classes):
    # if gt_cls_cnt[i] > 0:
        # gt_cpb_all = np.vstack((gt_cpb_all, gt_cpb[i,0:gt_cls_cnt[i]] ))
# print(gt_cpb_all)

# print('\n gt_cpb : (idx, class, prob, y1, x1, y2, x2)', gt_cpb.shape, '\n')
# # print(gt_cls_cnt)
# # print(gt_cpb[3])        
        
        
        