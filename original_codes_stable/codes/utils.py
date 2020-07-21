import json
import os
import tensorflow as tf
from skimage.feature import peak_local_max
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import tensorflow.contrib.slim as slim
import sys
slim_models_path = '../modules/'
sys.path.append(slim_models_path)

#bbox generation config
rel_peak_thr = .3
rel_rel_thr = .3
ioa_thr = .6
topk_boxes = 3

def heat2bbox(heat_map, original_image_shape):
    
    h, w = heat_map.shape
    
    bounding_boxes = []

    heat_map = heat_map - np.min(heat_map)
    heat_map = heat_map / np.max(heat_map)

    bboxes = []
    box_scores = []

    peak_coords = peak_local_max(heat_map, exclude_border=False, threshold_rel=rel_peak_thr) # find local peaks of heat map

    heat_resized = cv2.resize(heat_map, (original_image_shape[1],original_image_shape[0]))  ## resize heat map to original image shape
    peak_coords_resized = ((peak_coords + 0.5) * 
                           np.asarray([original_image_shape]) / 
                           np.asarray([[h, w]])
                          ).astype('int32')

    for pk_coord in peak_coords_resized:
        pk_value = heat_resized[tuple(pk_coord)]
        mask = heat_resized > pk_value * rel_rel_thr
        labeled, n = ndi.label(mask) 
        l = labeled[tuple(pk_coord)]
        yy, xx = np.where(labeled == l)
        min_x = np.min(xx)
        min_y = np.min(yy)
        max_x = np.max(xx)
        max_y = np.max(yy)
        bboxes.append((min_x, min_y, max_x, max_y))
        box_scores.append(pk_value) # you can change to pk_value * probability of sentence matching image or etc.


    ## Merging boxes that overlap too much
    box_idx = np.argsort(-np.asarray(box_scores))
    box_idx = box_idx[:min(topk_boxes, len(box_scores))]
    bboxes = [bboxes[i] for i in box_idx]
    box_scores = [box_scores[i] for i in box_idx]

    to_remove = []
    for iii in range(len(bboxes)):
        for iiii in range(iii):
            if iiii in to_remove:
                continue
            b1 = bboxes[iii]
            b2 = bboxes[iiii]
            isec = max(min(b1[2], b2[2]) - max(b1[0], b2[0]), 0) * max(min(b1[3], b2[3]) - max(b1[1], b2[1]), 0)
            ioa1 = isec / ((b1[2] - b1[0]) * (b1[3] - b1[1]))
            ioa2 = isec / ((b2[2] - b2[0]) * (b2[3] - b2[1]))
            if ioa1 > ioa_thr and ioa1 == ioa2:
                to_remove.append(iii)
            elif ioa1 > ioa_thr and ioa1 >= ioa2:
                to_remove.append(iii)
            elif ioa2 > ioa_thr and ioa2 >= ioa1:
                to_remove.append(iiii)

    for i in range(len(bboxes)): 
        if i not in to_remove:
            bounding_boxes.append({
                'score': box_scores[i],
                'bbox': bboxes[i],
                'bbox_normalized': np.asarray([
                    bboxes[i][0] / heat_resized.shape[1],
                    bboxes[i][1] / heat_resized.shape[0],
                    bboxes[i][2] / heat_resized.shape[1],
                    bboxes[i][3] / heat_resized.shape[0],
                ]),
            })
    
    return bounding_boxes

def img_heat_bbox_disp(image, heat_map, title='', en_name='', alpha=0.6, cmap='viridis', cbar='False', dot_max=False, bboxes=[], order=None, show=True):
    thr_hit = 1 #a bbox is acceptable if hit point is in middle 85% of bbox area
    thr_fit = .60 #the biggest acceptable bbox should not exceed 60% of the image
    H, W = image.shape[0:2]
    # resize heat map
    heat_map_resized = cv2.resize(heat_map, (H, W))

    # display
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(title, size=15)
    ax = plt.subplot(1,3,1)
    plt.imshow(image)
    if dot_max:
        max_loc = np.unravel_index(np.argmax(heat_map_resized, axis=None), heat_map_resized.shape)
        plt.scatter(x=max_loc[1], y=max_loc[0], edgecolor='w', linewidth=3)
    
    if len(bboxes)>0: #it gets normalized bbox
        if order==None:
            order='xxyy'
        
        for i in range(len(bboxes)):
            bbox_norm = bboxes[i]
            if order=='xxyy':
                x_min,x_max,y_min,y_max = int(bbox_norm[0]*W),int(bbox_norm[1]*W),int(bbox_norm[2]*H),int(bbox_norm[3]*H)
            elif order=='xyxy':
                x_min,x_max,y_min,y_max = int(bbox_norm[0]*W),int(bbox_norm[2]*W),int(bbox_norm[1]*H),int(bbox_norm[3]*H)
            x_length,y_length = x_max-x_min,y_max-y_min
            box = plt.Rectangle((x_min,y_min),x_length,y_length, edgecolor='w', linewidth=3, fill=False)
            plt.gca().add_patch(box)
            if en_name!='':
                ax.text(x_min+.5*x_length,y_min+10, en_name,
                verticalalignment='center', horizontalalignment='center',
                #transform=ax.transAxes,
                color='white', fontsize=15)
                #an = ax.annotate(en_name, xy=(x_min,y_min), xycoords="data", va="center", ha="center", bbox=dict(boxstyle="round", fc="w"))
                #plt.gca().add_patch(an)
            
    plt.imshow(heat_map_resized, alpha=alpha, cmap=cmap)
    
    #plt.figure(2, figsize=(6, 6))
    plt.subplot(1,3,2)
    plt.imshow(image)
    #plt.figure(3, figsize=(6, 6))
    plt.subplot(1,3,3)
    plt.imshow(heat_map_resized)
    fig.tight_layout()
    fig.subplots_adjust(top=.85)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def filter_bbox(bbox_dict,order=None):
    thr_fit = .90 #the biggest acceptable bbox should not exceed 80% of the image
    if order==None:
            order='xxyy'
        
    filtered_bbox = []
    filtered_bbox_norm = []
    filtered_score = []
    if len(bbox_dict)>0: #it gets normalized bbox
        for i in range(len(bbox_dict)):
            bbox = bbox_dict[i]['bbox']
            bbox_norm = bbox_dict[i]['bbox_normalized']
            bbox_score = bbox_dict[i]['score']
            if order=='xxyy':
                x_min,x_max,y_min,y_max = bbox_norm[0],bbox_norm[1],bbox_norm[2],bbox_norm[3]
            elif order=='xyxy':
                x_min,x_max,y_min,y_max = bbox_norm[0],bbox_norm[2],bbox_norm[1],bbox_norm[3]
            if bbox_score>0:
                x_length,y_length = x_max-x_min,y_max-y_min
                if x_length*y_length<thr_fit:
                    filtered_score.append(bbox_score)
                    filtered_bbox.append(bbox)
                    filtered_bbox_norm.append(bbox_norm)
    return filtered_bbox, filtered_bbox_norm, filtered_score

def load_model(model_path,config):
    new_graph = tf.Graph()
    sess = tf.InteractiveSession(graph = new_graph, config=config)
    new_saver = tf.train.import_meta_graph(model_path+'.meta')
    _ = sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    new_saver.restore(sess, model_path)
    return sess, new_graph

def crop_resize_im(image, bbox, size, order='xxyy'):
    H,W,_ = image.shape
    if order=='xxyy':
        roi = image[int(bbox[2]*H):int(bbox[3]*H),int(bbox[0]*W):int(bbox[1]*W),:]
    elif order=='xyxy':
        roi = image[int(bbox[1]*H):int(bbox[3]*H),int(bbox[0]*W):int(bbox[2]*W),:]
    roi = cv2.resize(roi,size)
    return roi
   
def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

def IoU(boxA, boxB):
    #order = xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou


class pre_trained_load():
    """
    Building a TF graph based on pre-trained image classification models.
    This implementation supports Slim.
    """
    def __init__(self, model_name, image_shape=(None,224,224,3),
                 input_tensor=None, session=None, is_training=True,
                 global_pool=False, num_classes=None):
        supported_list = ['vgg_16','InceptionV3','InceptionV4','pnasnet_large','resnet_v2_152','resnet_v2_101','resnet_v2_50']
        if model_name not in supported_list:
            raise ValueError('Provide a valid/supported model name.')
            return
        self.model_name = model_name
        self.image_shape = image_shape
        self.is_training = is_training
        self.global_pool = global_pool
        self.num_classes = num_classes
        self._build_graph(input_tensor)
        self.sess = session

    def _build_graph(self, input_tensor):
        with tf.name_scope('inputs'):
            if input_tensor is None:
                input_tensor = tf.placeholder(tf.float32, shape=self.image_shape, name='input_img')
            else:
                assert self.image_shape == tuple(input_tensor.shape.as_list())
            self.input_tensor = input_tensor

        if self.model_name == 'vgg_16':
            self.ckpt_path = slim_models_path+"vgg_16.ckpt"
            from nets.vgg import vgg_16, vgg_arg_scope
            with slim.arg_scope(vgg_arg_scope()):
                self.output, self.outputs = vgg_16(self.input_tensor, num_classes=self.num_classes, is_training=self.is_training, global_pool=self.global_pool)
                
        if self.model_name == 'resnet_v2_152':
            self.ckpt_path = slim_models_path+"resnet_v2_152.ckpt"
            from nets.resnet_v2 import resnet_v2_152, resnet_arg_scope
            with slim.arg_scope(resnet_arg_scope()):
                self.output, self.outputs = resnet_v2_152(self.input_tensor, num_classes=self.num_classes, is_training=self.is_training, global_pool=self.global_pool)
        
        elif self.model_name == 'resnet_v2_101':
            self.ckpt_path = slim_models_path+"resnet_v2_101.ckpt"
            from nets.resnet_v2 import resnet_v2_101, resnet_arg_scope
            with slim.arg_scope(resnet_arg_scope()):
                self.output, self.outputs = resnet_v2_101(self.input_tensor, num_classes=self.num_classes, is_training=self.is_training, global_pool=self.global_pool)
        
        elif self.model_name == 'resnet_v2_50':
            self.ckpt_path = slim_models_path+"resnet_v2_50.ckpt"
            from nets.resnet_v2 import resnet_v2_50, resnet_arg_scope
            with slim.arg_scope(resnet_arg_scope()):
                self.output, self.outputs = resnet_v2_50(self.input_tensor, num_classes=self.num_classes, is_training=self.is_training, global_pool=self.global_pool)
        
        elif self.model_name == 'InceptionV3':
            self.ckpt_path = slim_models_path+"inception_v3.ckpt"
            from nets.inception import inception_v3, inception_v3_arg_scope
            with slim.arg_scope(inception_v3_arg_scope()):
                self.output, self.outputs = inception_v3(self.input_tensor, num_classes=self.num_classes, is_training=self.is_training)
                
        elif self.model_name == 'InceptionV4':
            self.ckpt_path = slim_models_path+"inception_v4.ckpt"
            from nets.inception import inception_v4, inception_v4_arg_scope
            with slim.arg_scope(inception_v4_arg_scope()):
                self.output, self.outputs = inception_v4(self.input_tensor, num_classes=self.num_classes, is_training=self.is_training)
                
        elif self.model_name == 'pnasnet_large':
            self.ckpt_path = slim_models_path+"pnasnet_large_2.ckpt"
            from nets.nasnet.pnasnet import build_pnasnet_large, pnasnet_large_arg_scope
            with tf.variable_scope(self.model_name):
                with slim.arg_scope(pnasnet_large_arg_scope()):
                    self.output, self.outputs = build_pnasnet_large(self.input_tensor, num_classes=self.num_classes, is_training=self.is_training)
            
        #collecting all variables related to this model
        #self.model_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_name+'/model')
        self.model_weights = slim.get_model_variables(self.model_name)

    def load_weights(self):
        model_saver = tf.train.Saver(self.model_weights)
        model_saver.restore(self.sess, self.ckpt_path)

    def __getitem__(self, key):
        return self.outputs[key]
    
def pre_process(image, pre_processing_name):
    if pre_processing_name=='vgg_preprocessing':
        with tf.variable_scope(pre_processing_name):
            image = tf.subtract(image, [123.68,116.78,103.94])
        
    elif pre_processing_name=='inception_preprocessing':
        with tf.variable_scope(pre_processing_name):
            #image should be in range [-1,1]
            image = tf.divide(image, 255.0)
            image = tf.subtract(image, .5)
            image = tf.multiply(image, 2.0)
    else:
        raise ValueError('Provide a valid/supported model name.')
        return
    return image

def isCorrect(bbox_annot, bbox_pred, iou_thr=.4):
    for bbox_p in bbox_pred:
        for bbox_a in bbox_annot:
            if IoU(bbox_p,bbox_a)>=iou_thr:
                return 1
    return 0
def isCorrectHit(bbox_annot,heatmap,orig_img_shape):
    H,W = orig_img_shape
    heatmap_resized = cv2.resize(heatmap, (W, H))
    max_loc = np.unravel_index(np.argmax(heatmap_resized, axis=None), heatmap_resized.shape)
    for bbox in bbox_annot:
        if bbox[0]<=max_loc[1]<=bbox[2] and bbox[1]<=max_loc[0]<=bbox[3]:
            return 1
    return 0

def check_percent(bboxes):
    for bbox in bboxes:
        x_length = bbox[2]-bbox[0]
        y_length = bbox[3]-bbox[1]
        if x_length*y_length<.05:
            return False
    return True

def union(bbox):
    if len(bbox)==0:
        return []
    if type(bbox[0]) == type(0.0) or type(bbox[0]) == type(0):
        bbox = [bbox]
    maxes = np.max(bbox,axis=0)
    mins = np.min(bbox,axis=0)
    return [[mins[0],mins[1],maxes[2],maxes[3]]]

def attCorrectness(bbox_annot,heatmap,orig_img_shape):
    H,W = orig_img_shape
    heatmap_resized = cv2.resize(heatmap, (W, H))
    h_s = np.sum(heatmap_resized)
    if h_s==0:
        return 0
    else:
        heatmap_resized /= h_s
    att_correctness = 0
    for bbox in bbox_annot:
        x0,y0,x1,y1=bbox
        att_correctness+=np.sum(heatmap_resized[y0:y1,x0:x1])
    return att_correctness
    
def calc_correctness(annot,heatmap,orig_img_shape):
    bbox_dict = heat2bbox(heatmap,orig_img_shape)
    bbox, bbox_norm, bbox_score = filter_bbox(bbox_dict=bbox_dict, order='xyxy')
    bbox_norm_annot = union(annot['bbox_norm'])
    bbox_annot = union(annot['bbox'])
    bbox_norm_pred = union(bbox_norm)
    bbox_correctness = isCorrect(bbox_norm_annot, bbox_norm_pred, iou_thr=.5)
    hit_correctness = isCorrectHit(bbox_annot,heatmap,orig_img_shape)
    return bbox_correctness,hit_correctness

def img_heat_bbox_disp(image, heat_map, title='', en_name='', alpha=0.6, cmap='viridis', cbar='False', dot_max=False, bboxes=[], order=None, show=True):
    thr_hit = 1 #a bbox is acceptable if hit point is in middle 85% of bbox area
    thr_fit = .60 #the biggest acceptable bbox should not exceed 60% of the image
    H, W = image.shape[0:2]
    # resize heat map
    heat_map_resized = cv2.resize(heat_map, (H, W))

    # display
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(title, size=15)
    ax = plt.subplot(1,3,1)
    plt.imshow(image)
    if dot_max:
        max_loc = np.unravel_index(np.argmax(heat_map_resized, axis=None), heat_map_resized.shape)
        plt.scatter(x=max_loc[1], y=max_loc[0], edgecolor='w', linewidth=3)
    
    if len(bboxes)>0: #it gets normalized bbox
        if order==None:
            order='xxyy'
        
        for i in range(len(bboxes)):
            bbox_norm = bboxes[i]
            if order=='xxyy':
                x_min,x_max,y_min,y_max = int(bbox_norm[0]*W),int(bbox_norm[1]*W),int(bbox_norm[2]*H),int(bbox_norm[3]*H)
            elif order=='xyxy':
                x_min,x_max,y_min,y_max = int(bbox_norm[0]*W),int(bbox_norm[2]*W),int(bbox_norm[1]*H),int(bbox_norm[3]*H)
            x_length,y_length = x_max-x_min,y_max-y_min
            box = plt.Rectangle((x_min,y_min),x_length,y_length, edgecolor='w', linewidth=3, fill=False)
            plt.gca().add_patch(box)
            if en_name!='':
                ax.text(x_min+.5*x_length,y_min+10, en_name,
                verticalalignment='center', horizontalalignment='center',
                #transform=ax.transAxes,
                color='white', fontsize=15)
                #an = ax.annotate(en_name, xy=(x_min,y_min), xycoords="data", va="center", ha="center", bbox=dict(boxstyle="round", fc="w"))
                #plt.gca().add_patch(an)
            
    plt.imshow(heat_map_resized, alpha=alpha, cmap=cmap)
    
    #plt.figure(2, figsize=(6, 6))
    plt.subplot(1,3,2)
    plt.imshow(image)
    #plt.figure(3, figsize=(6, 6))
    plt.subplot(1,3,3)
    plt.imshow(heat_map_resized)
    fig.tight_layout()
    fig.subplots_adjust(top=.85)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig