import math
import random
import numpy as np
import cv2

from mrcnn.visualize import display_images
from mrcnn.dataset   import Dataset
# from mrcnn.shapes    import ShapesConfig
from mrcnn.datagen   import load_image_gt
from mrcnn.visualize import draw_boxes
from   mrcnn.config  import Config
from   mrcnn.dataset import Dataset
from mrcnn.utils     import non_max_suppression, mask_string
# import mrcnn.utils as utils
import pprint
p4 = pprint.PrettyPrinter(indent=4, width=100)
p8 = pprint.PrettyPrinter(indent=8, width=100)

 


class NewShapesConfig(Config):
    '''
    Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    '''
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 6 shapes
    SHAPES_PER_IMAGE = 7

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


    
class NewShapesDataset(Dataset):
    '''
    Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    '''
    # def __init__(self, height, width ):
        # self.height = height
        # self.width  = width 
    
    
    def load_shapes(self, count, height, width, shapes_per_image=7, buffer = 20):
        '''
        Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        '''
        
        # Add classes
        # self.add_class("shapes", 1, "circle")  # used to be class 2
        # self.add_class("shapes", 2, "square")  # used to be class 1
        # self.add_class("shapes", 3, "triangle")
        # self.add_class("shapes", 4, "rectangle")
        self.add_class("shapes", 1, "person")
        self.add_class("shapes", 2, "car")
        self.add_class("shapes", 3, "sun")
        self.add_class("shapes", 4, "building")
        self.add_class("shapes", 5, "tree")
        self.add_class("shapes", 6, "cloud")
        self.buffer = buffer
        self.shapes_per_image = shapes_per_image
        print(' Shapes Per Image: ', self.shapes_per_image)
        self.Min_Y = {}
        self.Max_Y = {}
        self.Min_X = {}
        self.Max_X = {}
        
        self.Min_X['_default'] =  buffer
        self.Max_X['_default'] =  height - buffer - 1
        
        self.Min_X['person']   =  buffer
        self.Max_X['person']   =  height - buffer - 1
        self.Min_X['car'   ]   =  buffer
        self.Max_X['car'   ]   =  height - buffer - 1
        self.Min_X['building'] =  buffer
        self.Max_X['building'] =  height - buffer - 1
        self.Min_X['sun']      =  buffer //3                
        self.Max_X['sun']      =  width - (buffer//3) - 1  
        self.Min_X['tree']     =  buffer
        self.Max_X['tree']     =  height - buffer - 1
        self.Min_X['cloud']    =  buffer//2                 
        self.Max_X['cloud']    =  width - (buffer//2) - 1    
        
        
        self.Min_Y['_default'] =  buffer
        self.Max_Y['_default'] =  height - buffer - 1
        
        self.Min_Y['person']   =  height //2 
        self.Max_Y['person']   =  height - buffer - 1
        self.Min_Y['car'   ]   =  height //2
        self.Max_Y['car'   ]   =  height - buffer - 1
        self.Min_Y['building'] =  height //3 
        self.Max_Y['building'] =  2 * height //3   ##* min_range_y
        self.Min_Y['sun'   ]   =  buffer //3
        self.Max_Y['sun'   ]   =  height //5    ##* min_range_y
        self.Min_Y['tree'  ]   =  height // 3
        self.Max_Y['tree'  ]   =  width - (buffer) - 1    ##* min_range_y
        self.Min_Y['cloud' ]   =  buffer
        self.Max_Y['cloud' ]   =  height //4
        
        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            # if i % 25 == 0:
                # print(' Add image ---> ',i )
            bg_color, shapes = self.random_image(i, height, width)
            
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        '''
        Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but in this case it
        generates the image on the fly from the specs in image_info.
        '''
        # print(' ===> Loading image * image_id : ',image_id)
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        # print(" Load Image : Shapes ")
        # p4.pprint(info['shapes'])

        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)        
        return image

    
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)    

            
    def load_mask(self, image_id):
        '''
        Generate instance masks for shapes of the given image ID.
        '''
        # print(' ===> Loading mask info for image_id : ',image_id)
        info   = self.image_info[image_id]
        shapes = info['shapes']
        
        # print('\n Load Mask information (shape, (color rgb), (x_ctr, y_ctr, size) ): ')
        # p4.pprint(info['shapes'])
        count  = len(shapes)
        mask   = np.zeros([info['height'], info['width'], count], dtype=np.uint8)

        # print(' Shapes obj mask shape is :',mask.shape)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(), shape, dims, 1)
        
        #----------------------------------------------------------------------------------
        ## Handle occlusions 
        #   Occlusion starts with the last object an list and in each iteration of the loop 
        #   adds an additional  object. Pixes assigned to objects are 0. Non assigned pixels 
        #   are 1
        #-----------------------------------------------------------------------------------
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            # print('------------------------------------')
            # print(' i is :', i, 'BEFORE Mask - shape: ', mask[:, :, i:i + 1].shape, ' Mask all zeros: ', ~np.any(mask[:, :, i:i + 1]))
            # print('------------------------------------')            
            # print(mask_string(mask[:,:,i]))
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        
        # Assign class Ids to each shape --- Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)
    
    def find_hidden_shapes(self, shapes, height, width):
        '''
        A variation of load_masks customized to find objects that 
        are completely hidden by other shapes 
        '''

        # print('\n Load Mask information (shape, (color rgb), (x_ctr, y_ctr, size) ): ')
        # p4.pprint(info['shapes'])
        hidden_shapes = []
        count  = len(shapes)
        mask   = np.zeros( [height, width, count], dtype=np.uint8)

        ## get masks for each shape 
        for i, (shape, _, dims) in enumerate(shapes):
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(), shape, dims, 1)
        
        #----------------------------------------------------------------------------------
        #  Start with last shape as the occlusion mask
        #   Occlusion starts with the last object an list and in each iteration of the loop 
        #   adds an additional  object. Pixes assigned to objects are 0. Non assigned pixels 
        #   are 1
        #-----------------------------------------------------------------------------------
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)

        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

            ##-------------------------------------------------------------------------------------
            ## if the shape has been completely occluded by other shapes, it's mask is all zeros.
            ## in this case np.any(mask) will return FALSE.
            ## For these completely hidden objects, we record their id in hidden [] 
            ## and later remove them from the  list of shapes 
            ##-------------------------------------------------------------------------------------
            if ( ~np.any(mask[:,:,i]) ) :
                # print(' !!!!!!  Zero Mask Found !!!!!!' )
                hidden_shapes.append(i)

        # if len(hidden_shapes) > 0 :
            # print(' ===> Find Hidden Shapes() found hidden objects ')
            # p8.pprint(shapes)
            # print(' ****** Objects completely hidden are : ', hidden_shapes)
            # for i in hidden_shapes:
                # p8.pprint(shapes[i])
        return hidden_shapes
    
    
    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, sx, sy = dims
        # print(' draw_image() Shape : {:20s}   Cntr (x,y): ({:3d} , {:3d})    Size_x: {:3d}   Size_y: {:3d} {}'.format(shape,x,y,sx, sy,color))        
        
        if shape == "square":
            image = cv2.rectangle(image, (x - sx, y - sy), (x + sx, y + sy), color, -1)

        elif shape in ["rectangle", "building"]:
            image = cv2.rectangle(image, (x - sx, y - sy), (x + sx, y + sy), color, -1)        
#             print('X :', x, 'y:', y , '     sx: ',sx , 'sy: ', sy, 'hs:', hs)
        
        elif shape == "car":
            body_y = sy //3
            wheel_x = sx //2
            wheel_r = sx //5
            top_x   = sx //4
            bot_x   = 3*sx //4
            image = cv2.rectangle(image, (x - sx, y - body_y), (x + sx, y + body_y), color, -1)    
            image = cv2.circle(image, (x - wheel_x , y + body_y), wheel_r, color, -1)     
            image = cv2.circle(image, (x + wheel_x , y + body_y), wheel_r, color, -1)     

            points = np.array([[(x - top_x , y - sy),   (x + top_x, y - sy),
                                (x + bot_x,  y - body_y),(x - bot_x, y - body_y), ]], dtype=np.int32)
            image = cv2.fillPoly(image, points, color)                     
            
        elif shape == "person":
#             hy = sy // 4   # head height
#             by = sy - hy   # body height
#             print('X :', x, 'y:', y , 'sx: ',sx , 'sy: ', sy, 'hs:', hs)                         
#             image = cv2.rectangle(image, (x - sx, y - by), (x + sx, y + by), color, -1)    
#             image = cv2.circle(image, (x , y -(by+hy) ), sx, color, -1)            

            hy = sy // 4   # head height
            by = sy - hy   # body height
#             print('X :', x, 'y:', y , 'sx: ',sx , 'sy: ', sy, 'hs:', hs)            
            # torso
            image = cv2.rectangle(image, (x - sx, y - by), (x + sx, y + by//4), color, -1)    
            # legs
            image = cv2.rectangle(image, (x - sx, y + by//4), (x - sx +sx//4, y + by), color, -1)    
            image = cv2.rectangle(image, (x + sx - sx//4, y + by//4), (x + sx, y + by), color, -1)    
            #head
            image = cv2.circle(image, (x , y -(by+hy) ), sx, color, -1)           
        elif shape in ["circle", "sun"]:
            image = cv2.circle(image, (x, y), sx, color, -1)

        elif shape in ["cloud", "ellipse"]:
            image = cv2.ellipse(image,(x,y),(sx, sy),0,0,360,color,-1)            

        elif shape == "triangle":
            sin60 = math.sin(math.radians(60))
            # orde of points: top, left, right
            points = np.array([[(x, y - sx),
                                (x - (sx / sin60), y + sx),
                                (x + (sx / sin60), y + sx),
                                ]], dtype=np.int32)
            image = cv2.fillPoly(image, points, color)
        
        elif shape == "tree":
            sin60 = math.sin(math.radians(60))
            ty = sy //3            # trunk length
            by = sy - ty           # body length
            tx = int((by /sin60)//5)   # trunk width
        #     print('sx: ',sx , 'sy: ', sy, 'tx/ty :', tx, ' bx: ',bx)
            sin60 = math.sin(math.radians(60))
            # orde of points: top, left, right
            points = np.array([[(x, y - by),
                                (x - (by / sin60), y + by),
                                (x + (by / sin60), y + by),
                                ]], dtype=np.int32)
            image = cv2.fillPoly(image, points, color)             
            image = cv2.rectangle(image,(x-tx,y+by), (x+tx, y+by+ty),color, -1)                      
            
        return image            

        
    def random_shape(self, shape, height, width):
        '''
        Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * color:     Shape color: a tuple of 3 values, RGB.
        * x,y  :     location od center of object
        * sx,sy:     Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        '''
        # Shape
#         shape = random.choice(["square", "circle", "triangle", "rectangle", "person", "car"])
        
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        
        buffer      = self.buffer
        min_range_x = self.Min_X[shape]  
        max_range_x = self.Max_X[shape]  
        min_range_y = self.Min_Y[shape]
        max_range_y = self.Max_Y[shape]

        ## get random center of object (which is constrainted by min/mx ranges above)
        x = random.randint(min_range_x, max_range_x)
        y = random.randint(min_range_y, max_range_y)
        
        if shape == "person":

            min_height = 10
            max_height = 20
#             sy = random.randint(min_height, max_height)
            sy = int(np.interp([y],[min_range_y,  max_range_y], [min_height, max_height]))
            sx = sy //5    # body width 

        elif shape == "car":

            min_width = 15
            max_width = 30 
            sx = int(np.interp([y],[min_range_y, max_range_y], [min_width, max_width]))

            # scale width based on location on the image. Images closer to the bottom will be larger
            # old method
            # sx = random.randint(min_width , max_width)            
            sy = sx //2            
            
        elif shape == "building":            

            min_height = 10
            max_height = 30
            sy = int(np.interp([y],[min_range_y, max_range_y], [min_height, max_height]))
            #     sy = random.randint(min_height, max_height)
            #     sx = random.randint(5,15)
            sx = sy //2 + 5            
            
        elif shape == "sun":

            min_height = 4
            max_height = 10
            sx = int(np.interp([y],[min_range_y, max_range_y], [min_height, max_height]))
#             sx = random.randint(min_height, max_height)            
            sy = sx

        elif shape == "tree":

            min_height = 8
            max_height = 24
            sy = int(np.interp([y],[min_range_y, max_range_y], [min_height, max_height]))    
            #     sy = random.randint(min_height, max_height)            
            sx = sy
                      
        elif shape == "cloud":               

            min_width = 15 
            max_width = 40 
            sx = int(np.interp([y],[min_range_y, max_range_y], [min_width, max_width]))
        #     min_height ,max_height = 10, 20
        #     sy = random.randint(min_height, max_height)            
            sy = sx //  random.randint(3, 5)
                    
        else :
            min_width   = buffer
            max_width   = width // 4

            x = random.randint(self.Min_X['_default'], self.Max_X['_default'])
            y = random.randint(self.Min_Y['_default'], self.Max_Y['_default'])
            sx = int(np.interp([y],[self.Min_Y['_default'], self.Max_Y['_default']], [min_width, max_width]))    
            
            # sx = random.randint(min_size, max_size)

            if shape == "rectangle":
                sy = random.randint(min_size, max_size)    
            else:
                ## other shapes have same sx and sy             
                sy = sx
                
         

        return  color, (x, y, sx, sy)
    
    
    def random_image(self, image_id, height, width):
        '''
        Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        '''
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])

        # Generate a few random shapes and record their
        # bounding boxes
        shapes     = []
        N = random.randint(1, self.shapes_per_image)    # number to shapes in image 
        
        shape_choices = ["person", "car", "sun", "building", "tree", "cloud"]
        
        for _ in range(N):
            shape = random.choice(shape_choices)            
            color, dims = self.random_shape(shape, height, width)
            shapes.append((shape, color, dims))
            if shape == "sun":
                shape_choices.remove("sun")
            # following two lines have been moved below, after removal of hidden_shapes    
            # x, y, sx, sy = dims
            # boxes.append([y - sy, x - sx, y + sy, x + sx])

            
        ##--------------------------------------------------------------------------------
        ## Reorder shapes by increasing cy to simulate overlay 
        ## (nearer shapes cover farther away shapes)
        # order shape objects based on closeness to bottom of image (-1) or top (+1)
        # this will result in items closer to the viewer have higher priority in NMS
        #--------------------------------------------------------------------------------
        sort_lst = [itm[2][1] for itm in shapes]
        sorted_shape_ind = np.argsort(np.array(sort_lst))[::+1]

        # print(" =====  Before final sort =====  ")
        # p4.pprint(shapes)
        # print(sort_lst)
        # print(sorted_shape_ind)
        tmp_shapes = []
        for i in sorted_shape_ind:
            tmp_shapes.append(shapes[i])
        shapes = tmp_shapes        
        # print(' ===== Sahpes after sorting ===== ')
        # p4.pprint(shapes)

            
        ##-------------------------------------------------------------------------------
        ## find and remove shapes completely covered by other shapes 
        ##-------------------------------------------------------------------------------
        hidden_shape_ixs = self.find_hidden_shapes(shapes, height, width)    
        if len(hidden_shape_ixs) > 0: 
            non_hidden_shapes = [s for i, s in enumerate(shapes) if i not in hidden_shape_ixs]
            # print('    ===> Image Id : (',image_id, ')   ---- Zero Mask Encountered ') 
            # print('    ------ Original Shapes ------' )
            # p8.pprint(shapes)
            # print('    ------ shapes after removal of totally hidden shapes ------' )
            # p8.pprint(non_hidden_shapes)
            # print('    Number of shapes now is : ', len(non_hidden_shapes))
        else: 
            non_hidden_shapes = shapes

        ##-------------------------------------------------------------------------------
        ## build boxes for to pass to non_max_suppression
        ##-------------------------------------------------------------------------------
        boxes      = []
        for shp in non_hidden_shapes:
            x, y, sx, sy = shp[2]
            boxes.append([y - sy, x - sx, y + sy, x + sx])            
            
        ##--------------------------------------------------------------------------------
        ## Non Maximal Suppression
        ##--------------------------------------------------------------------------------
        # Suppress occulsions more than 0.3 IoU    
        # Apply non-max suppression with 0.3 threshold to avoid shapes covering each other
        # object scores (which dictate the priority) are assigned in the order they were created
        assert len(boxes) == len(non_hidden_shapes), "Problem with the shape and box sizes matching"
        N = len(boxes)
        
        keep_ixs =  non_max_suppression(np.array(boxes), np.arange(N), 0.29)        
        shapes = [s for i, s in enumerate(non_hidden_shapes) if i in keep_ixs]
        # print('===> Original number of shapes {}  # after NMS {}'.format(N, len(shapes)))
        

        #--------------------------------------------------------------------------------
        ## Reorder shapes to simulate overlay (nearer shapes cover farther away shapes)
        # order shape objects based on closeness to bottom of image (-1) or top (+1)
        # this will result in items closer to the viewer have higher priority in NMS
        #--------------------------------------------------------------------------------
        sort_lst = [itm[2][1] for itm in shapes]
        sorted_shape_ind = np.argsort(np.array(sort_lst))[::+1]

        # print(" =====  Before final sort =====  ")
        # p4.pprint(shapes)
        # print(sort_lst)
        # print(sorted_shape_ind)
        tmp_shapes = []
        for i in sorted_shape_ind:
            tmp_shapes.append(shapes[i])
        shapes = tmp_shapes        
        # print(' ===== Sahpes after sorting ===== ')
        # p4.pprint(shapes)

        return bg_color, shapes    
    
    
from mrcnn.utils import compute_iou
def debug_non_max_suppression(boxes, scores, threshold):
    '''
    Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    '''
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
      
    print(' ==== non_max_suppression ')
    print(boxes)
    print(scores)
    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)
    
    # Get indicies of boxes sorted by scores (highest first)
    
    ixs = scores.argsort()[::-1]
  
    pick = []
    print('====> Initial Ixs: ', ixs)
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        cy = y1[i] + (y2[i]-y1[i])//2
        cx = x1[i] + (x2[i]-x1[i])//2
        print('     ix : ', ixs, 'ctr (x,y)', cx,' ',cy,' box:', boxes[i], ' compare ',i, ' with ', ixs[1:])
        pick.append(i)
        print('     area[i]: ', area[i], 'area[ixs[1:]] :',area[ixs[1:]] )   
        # Compute IoU of the picked box with the rest
        iou = debug_compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        print('     ious:', iou)
        
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        tst =  np.where(iou>threshold)
        remove_ixs = np.where(iou > threshold)[0] + 1
        print('     np.where( iou > threshold) : ' ,tst, 'tst[0] (index into ixs[1:]: ', tst[0], 
         ' remove_ixs (index into ixs) : ',remove_ixs)
        
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
        print(' ending ixs (after deleting ixs[0]): ', ixs, ' picked so far: ',pick)
    print('====> Final Picks: ', pick)
    return np.array(pick, dtype=np.int32)

    
    
def debug_compute_iou(box, boxes, box_area, boxes_area):
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

    print('      box is :', box)
    print('      box[0]: ', box[0],'  boxes[:,0] : ', boxes[:,0], ' y1 - np.max ', y1)
    print('      box[2]: ', box[2],'  boxes[:,2] : ', boxes[:,2], ' y2 - np.min ', y2)
    print('      box[1]: ', box[1],'  boxes[:,1] : ', boxes[:,1], ' x1 - np.max ', x1)
    print('      box[3]: ', box[3],'  boxes[:,3] : ', boxes[:,3], ' x2 - np.min ', x2)
    
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    print('      intersection: ', intersection)
    union = box_area + boxes_area[:] - intersection[:]
    print('      union:        ' , union)
    iou = intersection / union  
    return iou

    