from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib

import argparse
import numpy as np
import glob

from matplotlib.widgets import Button
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.mlab import dist_point_to_segment

class PolygonInteractor(object): # from matplotlib docs
    """
    An polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly

        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True)
        self.ax.add_line(self.line)
        #self._update_line(poly)

        cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        self.draw = canvas.mpl_connect('draw_event', self.draw_callback)
        self.clicking = canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.keypress = canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.clickrelease = canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.motionevent = canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'

        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == 'd':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = [tup for i, tup in enumerate(self.poly.xy) if i != ind]
                self.line.set_data(zip(*self.poly.xy))
        elif event.key == 'i':
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.array(
                        list(self.poly.xy[:i]) +
                        [(event.xdata, event.ydata)] +
                        list(self.poly.xy[i:]))
                    self.line.set_data(zip(*self.poly.xy))
                    break

        self.canvas.draw()

    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def unlink_editor(self):
        
        self.canvas.mpl_disconnect(self.draw)
        self.canvas.mpl_disconnect(self.keypress)
        self.canvas.mpl_disconnect(self.clicking)
        self.canvas.mpl_disconnect(self.clickrelease)
        self.canvas.mpl_disconnect(self.motionevent)

class COCO_dataset_generator(object): # from matplotlib docs
 
    def __init__(self, fig, ax):
        
        self.ax = ax 
        self.fig = fig
        self.polys = []
        self.zoom_scale, self.points, self.objects, self.prev, self.current_actors, self.submit_p = 1.2, [], [], None, [], None
        
        self.zoom_id = fig.canvas.mpl_connect('scroll_event', self.zoom)
        self.click_id = fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        axsubmit = plt.axes([0.7, 0.05, 0.1, 0.05])
        axreset = plt.axes([0.81, 0.05, 0.1, 0.05])
        self.b_reset = Button(axreset, 'Reset')
        self.b_reset.on_clicked(self.reset)
        self.b_submit = Button(axsubmit, 'Submit')
        self.b_submit.on_clicked(self.submit)
        
    def points_to_polygon(self, points):
        #points.extend([points[0], points[1]])
        return np.reshape(np.array(points), (len(points)/2, 2))

    def onclick(self, event):
        
        if event.button==1:
            self.points.extend([event.xdata, event.ydata])
        else:
            if len(self.points)>5:
                self.fig.canvas.mpl_disconnect(self.click_id)
                self.points.extend([self.points[0], self.points[1]])
                #self.prev.remove()
        
        if (len(self.points)>2):
            self.ax.plot([self.points[-4], self.points[-2]], [self.points[-3], self.points[-1]], 'b--')
        circle = plt.Circle((event.xdata,event.ydata),2.5,color='black')
        self.ax.add_artist(circle)
        self.fig.canvas.draw()
        
        if len(self.points)>2:
            if self.prev:
                self.prev.remove()
            self.p = PatchCollection([Polygon(self.points_to_polygon(self.points), closed=True)], facecolor='red', linewidths=0, alpha=0.4)
            self.ax.add_collection(self.p)
            self.prev = self.p
        
            self.fig.canvas.draw()
        
        if len(self.points)>4:
            print 'AREA OF POLYGON: ', self.find_poly_area(self.points)
        #print event.x, event.y       

    def find_poly_area(self, points):
        even, odd = range(0, len(points), 2), range(1, len(points), 2)
        #print (even, odd)
        np_pts = np.array(points)
        x, y = np_pts[even], np_pts[odd]
        #print (x,y)
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))) #shoelace algorithm

    def zoom(self, event):

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location

        if event.button == 'down':
            # deal with zoom in
            scale_factor = 1 / self.zoom_scale
        elif event.button == 'up':
            # deal with zoom out
            scale_factor = self.zoom_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print event.button

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
        self.ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
        self.ax.figure.canvas.draw()

    def reset(self, event):
        self.p.remove()
        self.points = []
        
    def submit(self, event):  
        self.click_id = fig.canvas.mpl_connect('button_press_event', self.onclick)
        print self.points
        self.polys.append(Polygon(self.points_to_polygon(self.points), closed=True, color=np.random.rand(3), alpha=0.4, fill=True))
        if self.submit_p:   
            self.submit_p.remove()
        self.submit_p = PatchCollection(self.polys, cmap=matplotlib.cm.jet, alpha=0.4)
        self.ax.add_collection(self.submit_p)
        self.points = []

if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_dir", required=True, help="Path to the image dir")
    args = vars(ap.parse_args())

    fig = plt.figure(figsize=(14, 14))
    ax = plt.gca()
    plt.axis('off')
    
    index = 0
    img_dir = args['image_dir']
    if img_dir[-1]=='/':
        img_dir = img_dir[:-1]
    pics = glob.glob(args["image_dir"]+'/*')
    print pics
    image = plt.imread(pics[0])
    plt.imshow(image, aspect='auto')
    
    gen = COCO_dataset_generator(fig, ax)
    
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    fig.canvas.mpl_disconnect(zoom_id)
    fig.canvas.mpl_disconnect(click_id)
