import os
import numpy as np
import cv2
import glob
import xml.etree.cElementTree as ET

from hsv_segmentor import return_hsv_mask

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Mask R-CNN on Custom Bags Dataset.')
parser.add_argument('--bags_dir', required=False, metavar="/path/to/bags/folder", help="Path to bags dataset folder")
parser.add_argument('--background_dir', required=False, default='background/', metavar="/path/to/background/images", help='Background images directory (default=background/)')
args = parser.parse_args()

count, shelf = 0, np.zeros(4)
rects = []

#def get_bbox(image, mask):
def find_coords(event,x,y,flags,param):
    
    global count, shelf, rects
    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        if count==0:
            shelf[0]+=y
            shelf[2]+=x
        elif count==1:
            shelf[1]=y
            shelf[2]+=x
            shelf[2]/=2
        elif count==2:
            shelf[1] = max(shelf[1], y)
            shelf[3]+=x
        elif count==3:
            shelf[3]+=x
            shelf[0]+=y
            shelf[0]/=2
            shelf[3]/=2
        count+=1
        
    if count==4:
        count=0
        rects.append(shelf)
        shelf = np.zeros(4)
        

def main():

    global rects
    
    image_paths = []
    classes = ('meixuan_brown_handbag', 'black_plain_bag', 'white_bag', 'mk_brown_wrislet', 'black_backpack', 'sm_bdrew_grey_handbag', 'black_ameligalanti', 'wine_red_handbag', 'nine_west_bag', 'sm_peach_backpack', 'sm_bclarre_blush_crossbody', 'lmk_brown_messenger_bag')
    
    
    
    for f in glob.glob(args.background_dir+'/*'):
        if 'txt' not in f:
            if not os.path.exists(f[:-3]+'txt'):
                
                print ('Please select the points using left click in <bottom left, topleft, top right and bottom right> order for each and every shelf in the image and press any keyboard key when done!')

                cv2.namedWindow('frame')
                image = cv2.imread(f)
                cv2.imshow('frame', image)
                cv2.setMouseCallback('frame',find_coords)
                
                cv2.waitKey(0)
                
                rects = np.transpose(np.array(rects, dtype=int))
                np.savetxt(f[:-3]+'txt', rects)
                
            else:
                rects = np.loadtxt(f[:-3]+'txt')
        
            bottom_ys, top_ys, start_xs, end_xs = rects
            
            rects = []
        
            x_ranges = np.array([[x1, x2] for x1, x2 in zip(start_xs, end_xs)])
            y_ranges = np.array([[y1, y2] for y1, y2 in zip(top_ys, bottom_ys)])
            
            ranges = np.array([[x,y] for x,y in zip(x_ranges, y_ranges)])
            
            for [x_range,y_range] in ranges:
                
                width, height = x_range[1]-x_range[0], y_range[1]-y_range[0]
                end = x_range[0]
                aspect = width/height
                
                while(end<x_range[1]-height):
                    
                    img_ht = height
                    img_wd = aspect*height
                    
                    if (end+img_wd>x_range[1]):
                        break
                    
                    
                    
                    end+=img_wd+5
                    
                print (x_range, y_range)
                print (width, height)
            
        cv2.destroyAllWindows()

        '''bg1.jpg
        bottom_ys = [308, 454, 605, 755]
        top_ys = [174, 340, 483, 634]
        start_xs = [377, 387, 390, 531]
        end_xs = [1376, 1376, 1370, 1371]
        '''
        
        '''bg2.jpg
        bottom_ys = [112, 165, 266]
        top_ys = [47, 128, 182]
        start_xs = [23, 23, 41]
        end_xs = [420, 423, 430]
        '''
        
        '''bg3.jpg
        bottom_ys = [315, 582, 840]
        top_ys = [85, 390, 648]
        start_xs = [212, 205, 215]
        end_xs = [1040, 1040, 1020]
        '''
        
        '''
        split = image_path.split('JPEGImages')
        annot_path = split[0]+'Annotations'+split[1][:-3]+'xml'
        
        #print (image_path, annot_path)
        
        tree = ET.parse(annot_path)
        root = tree.getroot()         
        
        num_masks, width, height = len(root.findall('object')), int(root.find('size').find('width').text), int(root.find('size').find('height').text)
        
        #print (height, width)
        
        image_masks, classes = np.zeros((height, width, num_masks), dtype=np.uint8), []
        
        for i, obj in enumerate(root.findall('object')):
        
            cls = obj.find('name').text
            bx = [int(obj.find('bndbox').find('ymin').text), int(obj.find('bndbox').find('xmin').text), int(obj.find('bndbox').find('ymax').text), int(obj.find('bndbox').find('xmax').text)]
            image_masks[:,:,i] = return_hsv_mask(image_path, np.array(bx), cls)
            classes.append(cls)
       
        mask = return_hsv_mask(image, np.array([0,0,image.shape[0],image.shape[1]])
        '''

if __name__=='__main__':
    main()
    
