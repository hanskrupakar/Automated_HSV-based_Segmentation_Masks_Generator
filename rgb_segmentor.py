from PIL import Image
import glob
import numpy as np
import cv2
import os
import math
from cv2 import moveWindow
from scipy import ndimage
from collections import Counter
import xml.etree.cElementTree as ET  
import time

from matplotlib import patches
import matplotlib.pyplot as plt

#fig = plt.figure()

def return_rgb_mask(image_path, bbox, bag_type):
    
    '''
    image_path: Location to image file
    bbox: Numpy array of [ymin, xmin, ymax, xmax]
    bag_type: Class of Bbox
    
    Returns:
    mask: Image segmentation mask based on color contrast in height*width Numpy array
    '''
    orig_im = im = Image.open(image_path)
    im = Image.fromarray(np.array(im)[bbox[0]:bbox[2], bbox[1]:bbox[3]], 'RGB')
    
    #orig_im.show()
    #im.show()
    
    mask = np.zeros((im.size[1], im.size[0]), dtype=np.uint8)
    sorte = im.getcolors(im.size[0]*im.size[1])
    sorte.sort(reverse=True, key= lambda x: x[0])

    one = np.array(sorte[0][1])
    two = np.array(sorte[1][1])
    dif = sum(abs(one-two))
    
    for n, col in sorte:
        if (sum(abs(one-col))>dif):
            two = col
            dif = sum(abs(one-col))
    
    if(sum(abs(one-np.array(im)[0][0]))<sum(abs(two-np.array(im)[0][0]))):
        temp = one
        one = two
        two = temp
    
    threshold = 650
    
    if ('red' in bag_type):
        threshold = 600
    elif ('white' in bag_type):
        threshold = 730
    elif ('peach' in bag_type):
        threshold = 575
    
    for key1, vals in enumerate(np.array(im)):
        for key2, rgb in enumerate(vals):
            if (key1<mask.shape[0] and key2<mask.shape[1]):
                if(sum(abs(rgb-one))<threshold):
                    mask[key1][key2]=255
                
    #print (np.count_nonzero(mask)/(im.size[0]*im.size[1]))
    
    '''
    if ((np.count_nonzero(mask)/(orig_im.size[0]*orig_im.size[1]))<0.025):
        mask = np.ones((im.size[1], im.size[0]), dtype=np.uint8)*255
    '''
    
    #print (mask.shape)
    height, width = orig_im.size[:2]
    #print (height, width)
    
    #Image.fromarray(mask, 'L').show()
    
    mask = np.lib.pad(mask, ((bbox[0],width-bbox[2]), (bbox[1],height-bbox[3])), 'constant', constant_values=0)
    
    return mask    

def get_rgb_masks(image_path):
    
    split = image_path.split('JPEGImages')
    
    annot_path = split[0]+'Annotations'+split[1][:-3]+'xml'
    
    #print (annot_path)
    
    tree = ET.parse(annot_path)
    root = tree.getroot()         
    
    num_masks, width, height = len(root.findall('object')), int(root.find('size').find('width').text), int(root.find('size').find('height').text)
    
    image_masks, classes = np.zeros((height, width, num_masks), dtype=np.uint8), []
    
    for i, obj in enumerate(root.findall('object')):
    
        cls = obj.find('name').text
        bx = [int(obj.find('bndbox').find('ymin').text), int(obj.find('bndbox').find('xmin').text), int(obj.find('bndbox').find('ymax').text), int(obj.find('bndbox').find('xmax').text)]
        image_masks[:,:,i] = return_mask(image_path, np.array(bx), cls)
        classes.append(cls)
    '''
    orig_im = Image.open(image_path)
    
    for i in range(image_masks.shape[-1]):
        mask = Image.fromarray(np.dstack((image_masks[...,i], image_masks[...,i], image_masks[...,i])), 'RGB')
        Image.blend(orig_im, mask, 0.75).show()
    '''
    return image_masks, classes

def test_time():
    tic = time.time()
    
    for f in glob.glob(os.getcwd()+'bags/white_bag/*.png'):
        return_mask(f, np.array([0, 0, 510, 510]), "")

    #seg, bb = find_object_bbox_masks(os.getcwd()+'/Data/bags/black_ameligalanti/2017-L7-CK2-20780452-01-1.jpg')

    #Image.fromarray(return_mask('/home/hans/Desktop/Vision Internship/github/Mask_RCNN/Data/handbag_images/JPEGImages/bot3.png', np.array([1, 1, 178, 192]), 'black_ameligalanti'), 'L').show()

    print (time.time()-tic)

if __name__=='__main__':
    test_time()
