"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import math
import random
import numpy as np
import tensorflow as tf
import scipy.misc
import skimage.color
import skimage.io


### Batch Slicing -------------------------------------------------------------------
##   Some custom layers support a batch size of 1 only, and require a lot of work
##   to support batches greater than 1. This function slices an input tensor
##   across the batch dimension and feeds batches of size 1. Effectively,
##   an easy way to support batches > 1 quickly with little code modification.
##   In the long run, it's more efficient to modify the code to support large
##   batches and getting rid of this function. Consider this a temporary solution
##   batch dimension size:
##       DetectionTargetLayer    IMAGES_PER_GPU  * # GPUs (batch size)
##-----------------------------------------------------------------------------------

def batch_slice(inputs, graph_fn, batch_size, names=None):
    """
    Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs:     list of tensors. All must have the same first dimension length
    graph_fn:   A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names:      If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]    # inputs is a list eg. [sc, ix] => input_slice = [sc[0], ix[0],...]
        output_slice = graph_fn(*inputs_slice)   # pass list of inputs_slices through function => graph_fn(sc[0], ix[0],...)
    
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    
    # Change outputs from:
    #    a list of slices where each is a list of outputs, e.g.  [ [out1[0],out2[0]], [out1[1], out2[1]],.....
    # to 
    #    a list of outputs and each has a list of slices ==>    [ [out1[0],out1[1],...] , [out2[0], out2[1],....],.....    
    
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]

def stack_tensors(x):
    ''' 
    stack an [Batch x Class x Row x Col] tensor into Row x Cols
    originally written for pred_tensor
    '''
    print(' input shape is : ', x.shape)
    lst2 = [ np.squeeze(item) for item in np.split( x, x.shape[0], axis = 0 )]
    lst2 = [ np.squeeze(np.concatenate(np.split(item, item.shape[0], axis = 0 ), axis = 1)) for item in lst2]
    result = [ item[~np.all(item[:,2:6] == 0, axis=1)] for item in lst2]
    print(' length of output list is : ', len(result))
    return (result)

def stack_tensors_3d(x):
    ''' 
    stack an  [Class x Row x Col] tensor into Row x Cols
    originally written for pred_tensor[img_id]
    ''' 
    print(' input shape is : ', x.shape)
    lst2   = [np.squeeze(item) for item in np.split( x, x.shape[0], axis = 0 )]
    result = np.concatenate( [ i[~np.all(i[:,2:6] == 0, axis=1)] for i in lst2] , axis = 0)
    print(' output shape is : ', result.shape)
    # print(result)
    return (result)


###############################################################################
## Data Formatting
###############################################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id    = meta[:, 0]
    image_shape = meta[:, 1:4]
    window      = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    """
    Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta:       [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id    = meta[:, 0]
    image_shape = meta[:, 1:4]
    window      = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding
	
###############################################################################
##  Bounding Boxes
###############################################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies   = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """
    Calculates IoU of the given box with the array of the given boxes.
    box:                1D vector [y1, x1, y2, x2]
    boxes:              [boxes_count, (y1, x1, y2, x2)]
    box_area:           float. the area of 'box'
    boxes_area:         array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union  
    return iou


def compute_overlaps(boxes1, boxes2):
    """
    Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """
    Applies the given deltas to the given boxes.
    boxes:          [N, (y1, x1, y2, x2)]. 
                    Note that (y2, x2) is outside the box.
    deltas:         [N, (dy, dx, log(dh), log(dw))]
    """
    
    boxes    = boxes.astype(np.float32)
    
    # Convert to y, x, h, w
    height   = boxes[:, 2] - boxes[:, 0]
    width    = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height   *= np.exp(deltas[:, 2])
    width    *= np.exp(deltas[:, 3])
    
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    
    return np.stack([y1, x1, y2, x2], axis=1)


###############################################################################
##  Miscellenous Graph Functions
###############################################################################

def trim_zeros_graph(boxes, name=None):
    """
    Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes by summing the coordinates
    of boxes, converting 0 to False and <> 0 ti True and creating a boolean mask
    
    boxes:      [N, 4] matrix of boxes.
    non_zeros:  [N] a 1D boolean mask identifying the rows to keep
    """
    # sum tf.abs(boxes) across axis 1 (sum all cols for each row) and cast to boolean.
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    
    # extract non-zero rows from boxes
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

def batch_pack_graph(x, counts, num_rows):
    """
    Picks different number of values from each row in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)
    
###############################################################################
## box_refinement_graph
###############################################################################
def box_refinement_graph(box, gt_box):
    """
    Compute refinement needed to transform box to gt_box.
    (tensorflow version)
    box and gt_box:     [N, (y1, x1, y2, x2)]
    """
    box         = tf.cast(box, tf.float32)
    gt_box      = tf.cast(gt_box, tf.float32)

    height      = box[:, 2] - box[:, 0]
    width       = box[:, 3] - box[:, 1]
    center_y    = box[:, 0] + 0.5 * height
    center_x    = box[:, 1] + 0.5 * width

    gt_height   = gt_box[:, 2] - gt_box[:, 0]
    gt_width    = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy          = (gt_center_y - center_y) / height
    dx          = (gt_center_x - center_x) / width
    dh          = tf.log(gt_height / height)
    dw          = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """
    Compute refinement needed to transform box to gt_box.
    (Non tensorflow version)
    box and gt_box:     [N, (y1, x1, y2, x2)]
                        (y2, x2) is  assumed to be outside the box.
    """
    box         = box.astype(np.float32)
    gt_box      = gt_box.astype(np.float32)

    height      = box[:, 2] - box[:, 0]
    width       = box[:, 3] - box[:, 1]
    center_y    = box[:, 0] + 0.5 * height
    center_x    = box[:, 1] + 0.5 * width

    gt_height   = gt_box[:, 2] - gt_box[:, 0]
    gt_width    = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy          = (gt_center_y - center_y) / height
    dx          = (gt_center_x - center_x) / width
    dh          = np.log(gt_height / height)
    dw          = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


###############################################################################
## Masks
###############################################################################

def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

    
def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
        mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().
    calls scipy resize to resize mask to the height and width of its corresponding bbox, 
    
    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
        mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = scipy.misc.imresize(
        mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


###############################################################################
## Pyramid Anchors
###############################################################################

def generate_anchors(scales, ratios, feature_shape, feature_stride, anchor_stride):
    '''
    scales:             1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios:             1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    feature_shape:      [height, width] spatial shape of the feature map over which
                        to generate anchors.
    feature_stride:     Stride of the feature map relative to the image in pixels.
    anchor_stride:      Stride of anchors on the feature map. For example, if the
                        value is 2 then generate anchors for every other feature map pixel.
    Returns
    -------
           Array of anchor box cooridnates in the format (y1,x1, y2,x2)
    '''
    
    # Get all combinations of scales and ratios
    # print('>>> generate_anchors()')
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    # print(' meshgrid scales and ratios: ' ,scales.shape, ratios.shape)
    
    scales = scales.flatten()
    ratios = ratios.flatten()
    
    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)  # 3x1
    widths  = scales * np.sqrt(ratios)  # 3x1

    # print(' flattened meshgrid scales and ratios: ' ,scales.shape, ratios.shape)    
    # print(' Heights ' ,heights, ' widths  ' ,widths)
    
    
    # Enumerate x,y shifts in feature space - which depends on the feature stride
    # for feature_stride 3 - shifts_x/y is 32
    # 
    shifts_y = np.arange(0, feature_shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, feature_shape[1], anchor_stride) * feature_stride
    # print(' Strides shift_x, shift_y:\n ' ,shifts_x,'\n', shifts_y)

    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    # print(' meshgrid shift_x, shift_y: ' ,shifts_x.shape, shifts_y.shape)
    
    # Enumerate combinations of shifts, widths, and heights
    # shape of each is [ shape[0] * shape[1] * size of (width/height)] 
    box_widths , box_centers_x = np.meshgrid(widths, shifts_x)    
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    
    # Reshape to get a list of (y, x) and a list of (h, w)
    # print(' box_widths  ', box_widths.shape ,' box_cneterss: ' , box_centers_x.shape)
    # print(' box_heights ', box_heights.shape,' box_cneters_y: ' , box_centers_y.shape)
    # print(' box_centers stack   :' , np.stack([box_centers_y, box_centers_x], axis=2).shape)
    # print(' box_centers reshape :' , np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1,2]).shape)
    # print(' box_sizes   stack   :' , np.stack([box_heights, box_widths], axis=2).shape)
    # print(' box_sizes   reshape :' , np.stack([box_heights, box_widths], axis=2).reshape([-1,2]).shape)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes   = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    # print(' Anchor boxes shape is : ' ,boxes.shape)
    return boxes


def generate_pyramid_anchors(anchor_scales, anchor_ratios, feature_shapes, feature_strides, anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    # print('\n>>> Generate pyramid anchors ')
    # print('      Anchor  scales:  ', anchor_scales)
    # print('      Anchor  ratios:  ', anchor_ratios)
    # print('      Anchor  stride:  ', anchor_stride)
    # print('      Feature shapes:  ', feature_shapes)
    # print('      Feature strides: ', feature_strides)

    anchors = []
    for i in range(len(anchor_scales)):
        anchors.append(generate_anchors(anchor_scales[i], anchor_ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    # anchors is a list of 5 np.arrays (one for each anchor scale)
    # concatenate these arrays on axis 0
    
    pp = np.concatenate(anchors, axis=0)
    # print('    Size of anchor array is :',pp.shape)
   
    return pp
   

###############################################################################
##  Miscellaneous
###############################################################################

def compute_ap(gt_boxes, gt_class_ids,
               pred_boxes, pred_class_ids, pred_scores,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding and sort predictions by score from high to low
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    overlaps = compute_overlaps(pred_boxes, gt_boxes)

    # Loop through ground truth boxes and find matching predictions
    match_count = 0
    pred_match = np.zeros([pred_boxes.shape[0]])
    gt_match = np.zeros([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] == 1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = 1
                pred_match[i] = 1
                break

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids



############################################################
#  Utility Functions
############################################################
def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


    