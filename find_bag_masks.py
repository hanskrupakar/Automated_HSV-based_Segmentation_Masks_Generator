import glob
import os
import sys
from rgb_segmentor import return_rgb_mask
from hsv_segmentor import get_bbox
import cv2
import numpy as np
import time
import xml.etree.cElementTree as ET
import shutil

def get_bgr_image(image_path, mask, color):
    
    '''
    Takes image path, it's binary mask and BGR color and gives the handbag with the background changed to color.
    '''
    
    img = cv2.imread(image_path)
    
    new_img = cv2.bitwise_and(img,img,mask = mask)
    new_img[np.where((new_img==[0,0,0]).all(axis=2))] = color
    
    return new_img

def main():
    
    full_path = sys.argv[1].split('/')
    if (sys.argv[1][-1]=='/'):
        full_path = full_path[:-1]
    
    #shutil.rmtree(os.getcwd()+'/'+'/'.join(full_path[:-1])+'/'+full_path[-1]+'_PASCAL', True)
    
    if (not os.path.isdir(os.getcwd()+'/'+'/'.join(full_path[:-1])+'/'+full_path[-1]+'_PASCAL')):
        os.mkdir('/'.join(full_path[:-1])+'/'+full_path[-1]+'_PASCAL')
    
    dataset_dir = os.getcwd()+'/'+'/'.join(full_path[:-1])+'/'+full_path[-1]+'_PASCAL'
    if (not os.path.isdir(dataset_dir+'/JPEGImages')):
        os.mkdir(dataset_dir+'/JPEGImages')
    if (not os.path.isdir(dataset_dir+'/Annotations')):
        os.mkdir(dataset_dir+'/Annotations')
    
    for img_path in glob.glob(sys.argv[1]+'/*/*.jpg'):
        
        cls = img_path.split('/')[-2]
        print (cls, os.path.exists(dataset_dir+'/JPEGImages/'+img_path.split('/')[-1]), dataset_dir+'/JPEGImages/'+img_path.split('/')[-1])
        
        if(not (os.path.exists(dataset_dir+'/JPEGImages/'+img_path.split('/')[-1]) and os.path.exists(dataset_dir+'/Annotations/'+img_path.split('/')[-1][:-3]+'xml'))):
            
            img = cv2.imread(img_path)
            mask = return_rgb_mask(img_path, np.array([0, 0, img.shape[0], img.shape[1]]), cls)
          
            bbox = get_bbox(img, mask)
            
            #sub_image = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            root = ET.Element("annotation")
            
            ET.SubElement(root, "path").text = os.getcwd()+'/'+img_path

            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(img.shape[1])
            ET.SubElement(size, "height").text = str(img.shape[0])
            ET.SubElement(size, "depth").text = str(img.shape[2])

            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, "name").text = cls
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(bbox[1])
            ET.SubElement(bndbox, "ymin").text = str(bbox[0])
            ET.SubElement(bndbox, "xmax").text = str(bbox[3])
            ET.SubElement(bndbox, "ymax").text = str(bbox[2])
            
            tree = ET.ElementTree(root)
            tree.write(dataset_dir+'/Annotations/'+img_path.split('/')[-1][:-3]+'xml')
            
            shutil.copy2(os.getcwd()+"/"+img_path, dataset_dir+'/JPEGImages/'+img_path.split('/')[-1])
        
            #new_img = get_bgr_image(img_path, mask, np.array([120,50,70]))
            
            '''
            cv2.imshow('frame', new_img)
            cv2.waitKey(0)
            cv2.imshow('image', img)
            #cv2.imshow('mask', return_hsv_mask(img_path, np.array([0, 0, img.shape[0], img.shape[1]]), cls))
            cv2.imshow('mask', mask)
            '''
        else:
            print ("BOO")

    cv2.destroyAllWindows()
    
if (__name__=='__main__'):
    main()
