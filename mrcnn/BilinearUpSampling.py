import keras.backend as KB
import tensorflow as tf
from keras.layers import *

def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    '''
    Resize the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default':
        data_format = KB.image_data_format()
    
    if data_format == 'channels_first':
        original_shape = KB.int_shape(X)

        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        
        X = permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = permute_dimensions(X, [0, 3, 1, 2])
    
        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X
    
    elif data_format == 'channels_last':
        original_shape = KB.int_shape(X)
        print('     CHANNELS LAST: X: ', X.get_shape(), ' KB.int_shape() : ', original_shape)
        print('     target_height   : ', target_height, ' target_width  : ', target_width )
        
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
            print('     new_shape (1): ' , new_shape.get_shape())            

        else:
            new_shape = tf.shape(X)[1:3]
            print('     new_shape (2): ' , new_shape.get_shape(), new_shape.shape)                        
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
            print('     new_shape (3): ' , new_shape.get_shape(), new_shape.shape)                        
        
        X = tf.image.resize_bilinear(X, new_shape)
        print('     X after image.resize_bilinear: ' , X.get_shape())            
        
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        print('     Dimensions of X after set_shape() : ', X.get_shape())
        return X
        
    else:
        raise Exception('Invalid data_format: ' + data_format)


class BilinearUpSampling2D(Layer):
    '''
    Deinfes the Bilinear Upsampling layer

    Returns:
    -------

    pred_tensor :       [batch, NUM_CLASSES, TRAIN_ROIS_PER_IMAGE    , (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
                                in normalized coordinates   
    pred_cls_cnt:       [batch, NUM_CLASSES] 
    gt_tensor:          [batch, NUM_CLASSES, DETECTION_MAX_INSTANCES, (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
    gt_cls_cnt:         [batch, NUM_CLASSES]
    
     Note: Returned arrays might be zero padded if not enough target ROIs.
    
    '''
    
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = KB.image_data_format()
            
        self.size = tuple(size)
        
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        print('\n>>> BilinearUpSampling2D layer' )
        print('     data_format : ', self.data_format)
        print('     size        : ', self.size   )
        print('     target_size : ', self.target_size)
        print('     input_spec  : ', self.input_spec)
        
        # super(BilinearUpSampling2D, self).__init__(**kwargs)
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        print('    BilinearUpSampling2D. compute_output_shape()' )    
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.data_format == 'channels_last':
            width  = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
            if self.target_size is not None:
                width  = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            print('     call resize_images_bilinear with target_size: ', self.target_size)
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1],
                                             data_format=self.data_format)
        else:
            print('     call resize_images_bilinear with size: ', self.size)        
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], 
                                             data_format=self.data_format)

    def get_config(self):
        print('    BilinearUpSampling2D. get_config()' )
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
