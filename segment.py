from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import argparse
import numpy as np

from matplotlib.widgets import Button

#GLOBALS

zoom_scale, points, objects, prev = 2, [], [], None

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
fig = plt.figure(figsize=(14, 14))
ax = plt.gca()
image = plt.imread(args["image"])

def points_to_polygon(points):
    #points.extend([points[0], points[1]])
    return np.reshape(np.array(points), (len(points)/2, 2))

def onclick(event):
    
    global prev, ax, fig, points
    #print points
    circle = plt.Circle((event.xdata,event.ydata),2.5,color='black')
    ax.add_artist(circle)
    fig.canvas.draw()
    
    if event.button==1:
        points.extend([event.xdata, event.ydata])
        
    else:
        points.extend([points[0], points[1]])
        print points
    
    if len(points)>2:
        if prev:
            prev.remove()
        p = PatchCollection([Polygon(points_to_polygon(points), closed=True)], facecolor='red', linewidths=0, alpha=0.4)
        ax.add_collection(p)
        prev = p
    
        fig.canvas.draw()
    
        print 'AREA OF POLYGON: ', find_poly_area(points)
    #print event.x, event.y       

def find_poly_area(points):
    even, odd = range(0, len(points), 2), range(1, len(points), 2)
    #print (even, odd)
    np_pts = np.array(points)
    x, y = np_pts[even], np_pts[odd]
    #print (x,y)
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))) #shoelace algorithm

def zoom(event):
    
    global zoom_scale, ax
    
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()

    xdata = event.xdata # get event x location
    ydata = event.ydata # get event y location

    if event.button == 'down':
        # deal with zoom in
        scale_factor = 1 / zoom_scale
    elif event.button == 'up':
        # deal with zoom out
        scale_factor = zoom_scale
    else:
        # deal with something that should never happen
        scale_factor = 1
        #print event.button

    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

    ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
    ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
    ax.figure.canvas.draw()

    fig = ax.get_figure() # get the figure of interest

if __name__=='__main__':

    plt.imshow(image, aspect='auto')
    zoom_id = fig.canvas.mpl_connect('scroll_event', zoom)
    click_id = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.subplots_adjust(bottom=0.2)
    plt.axis('off')
    plt.show()

    fig.canvas.mpl_disconnect(zoom_id)
    fig.canvas.mpl_disconnect(click_id)
