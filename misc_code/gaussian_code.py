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

    
def build_gaussian_np_2(in_tensor, in_cls_cnt, config):
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
    
    
    