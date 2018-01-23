'''
SYNTAX:

python find_hsv_value.py <IMAGE_PATH>

Loads the image and finds the maximum possible range of all the colours selected via mouse click on the image to generate image masks around the selected colour
'''
import cv2
import numpy as np
import sys

image_hsv = None   
pixel = (100,100,100) 

max_hsv, min_hsv = np.zeros(3), np.ones(3)*255

def find_mask(event,x,y,flags,param):
    
    global max_hsv, min_hsv
    
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]
        
        t1, t2 = 10, 40 #set these values to alter scaling of colours around colour at mouse-click
        upper =  np.array([pixel[0] + t1, pixel[1] + t1, pixel[2] + t2])
        lower =  np.array([pixel[0] - t1, pixel[1] - t1, pixel[2] - t2])
        
        for i in range(3): # find and track max possible range around multiple clicks
            if(max_hsv[i]<upper[i]):
                max_hsv[i]=upper[i]
            if(min_hsv[i]>lower[i]):
                min_hsv[i]=lower[i]

        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("mask",image_mask)
        print (max_hsv, min_hsv) #print the ranges


def main():

    global image_hsv, pixel # mouse callback

    image_src = cv2.imread(sys.argv[1])  

    if image_src is None:
        print ("File Error")
            
    cv2.imshow("bgr",image_src)

    cv2.namedWindow('hsv')
    cv2.setMouseCallback('hsv', find_mask)

    # now click into the hsv img , and look at values:
    image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv",image_hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
