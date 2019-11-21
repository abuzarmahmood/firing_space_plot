"""
=====================
Polygon Selector Demo
=====================

Shows how one can select indices of a polygon interactively.

"""
import numpy as np

from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import matplotlib.pyplot as plt
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import xarray as xr
from matplotlib.widgets import RadioButtons
import matplotlib.patches as patches


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `PolygonSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.poly = PolygonSelector(ax, self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

class NeuronCluster():

    """
    Plots scatter data as a hexbin scatter
    """

    def __init__(self, waveforms, embedded_data, nbins=100):
        if embedded_data.shape[1] > embedded_data.shape[0] \
                or not (embedded_data.shape[1] == 2):
            raise Exception('embedded_data must be of shape waveforms x 2') 
        elif not (waveforms.shape[0] == waveforms.shape[0]):
            raise Exception('Waveforms and embedded_data are not same length')
        else:
            self.embedded_data = embedded_data
            self.waveforms = waveforms
            self.nbins = nbins

    def find_closest(self, click_coords):
        return np.argmin(np.sum(np.abs\
                (self.embedded_data - np.asarray(click_coords)),axis=1))

#    def initiate_plots(self):
#        self.fig, self.ax = plt.subplots(nrows = 1, ncols = 2)
#        self.ax[0].hexbin(self.embedded_data[:,0],self.embedded_data[:,1])
#        self.ylims = \
#                self.ax[1].set_ylim([np.min(self.waveforms),np.max(self.waveforms)])
#        #fig, self.spike_ax  = plt.subplots()
#        self.cid = \
#                self.fig.canvas.mpl_connect('button_press_event', self.onclick)
#        plt.show()

    # Create array index identifiers
    # Used to convert array to pandas dataframe
    @staticmethod
    def make_array_identifiers(array):
        nd_idx_objs = []
        for dim in range(array.ndim):
            this_shape = np.ones(len(array.shape))
            this_shape[dim] = array.shape[dim]
            nd_idx_objs.append(
                    np.broadcast_to(
                        np.reshape(
                            np.arange(array.shape[dim]),
                                    this_shape.astype('int')), 
                        array.shape).flatten())
        return nd_idx_objs

    def make_waveform_array(self):
        idx = self.make_array_identifiers(self.waveforms)
        self.plot_waveform_frame = pd.DataFrame(\
                data = {'waveform': idx[0].flatten(),
                    'time' : idx[1].flatten(),
                    'voltage': self.waveforms.flatten()})

    def initiate_plots(self):
        self.fig = plt.figure(figsize = (6,4))
        self.ax = []
        self.ax.append(self.fig.add_axes([0.0,0.3,0.4,1]))
        self.ax.append(self.fig.add_axes([0.6,0.1,0.3,0.4]))
        self.ax.append(self.fig.add_axes([0.6,0.6,0.4,0.4]))
        self.ax.append(self.fig.add_axes([0.0,0.0,0.2,0.2], \
                facecolor  ='white'))
        self.ax[0].hexbin(self.embedded_data[:,0],self.embedded_data[:,1])
        self.this_spike, = self.ax[0].plot(self.embedded_data[0,0],\
                self.embedded_data[0,1],'o', c='red')

        self.make_waveform_array()
        plt.sca(self.ax[2])
        cvs = ds.Canvas(plot_height=400, plot_width=1000)
        agg = cvs.line(self.plot_waveform_frame, x='time', y = 'voltage',\
                agg = ds.count()) 
        img = tf.set_background(tf.shade(agg, how='eq_hist',cmap = 'lightblue'),\
                color = (1,1,1))
        img.plot()

        self.ylims = \
                self.ax[1].set_ylim([np.min(self.waveforms),np.max(self.waveforms)])
        self.ax[2].set_ylim(self.ylims)

        #plt.sca(self.ax[1])
        #self.waveform_plot = plt.plot(self.waveforms[0,:],'x-')
        self.waveform_plot, = self.ax[1].plot(self.waveforms[0,:],'x-')
        self.cid = \
                self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.radio_control = RadioButtons(self.ax[3],\
                ('Show','Cluster'))
        self.radio_control.on_clicked(self.color_change)
        #self.poly = PolygonSelector(self.ax[0], self.poly_clust)

        self.verts = []
        plt.show()

    def onclick(self,event):
        if event.inaxes == self.ax[0]:
            print('button_click')
            closest_spike = self.find_closest((event.xdata, event.ydata))
            print((closest_spike, event.xdata,event.ydata))
            self.waveform_plot.set_ydata(self.waveforms[closest_spike,:])
            self.this_spike.set_data([self.embedded_data[closest_spike,0]],\
                    [self.embedded_data[closest_spike,1]])
            self.fig.canvas.draw_idle()
            #plt.pause(0.01)
            #self.fig.canvas.mpl_connect(self.cid)

    def color_change(self, label):
        #self.poly_disconnect()
        col_dict = {'Show': self.onclick , 'Cluster': self.poly_clust}
        self.fig.canvas.mpl_disconnect(self.cid)
        self.cid = \
                self.fig.canvas.mpl_connect('button_press_event', \
                col_dict[label])

    def poly_clust(self, event):
        print('poly_select')
        self.verts.append((event.xdata, event.ydata))
        codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
            ]
        path = Path(self.verts)
        patch = patches.PathPatch(path, facecolor='orange', lw=2)
        self.ax[0].add_patch(patch)
        self.fig.canvas.draw()

    def poly_disconnect(self):
        self.poly.disconnect_events()
        #self.fc[:, -1] = 1
        #self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    grid_size = 5
    grid_x = np.tile(np.arange(grid_size), grid_size)
    grid_y = np.repeat(np.arange(grid_size), grid_size)
    pts = ax.scatter(grid_x, grid_y)

    selector = SelectFromCollection(ax, pts)

    print("Select points in the figure by enclosing them within a polygon.")
    print("Press the 'esc' key to start a new polygon.")
    print("Try holding the 'shift' key to move all of the vertices.")
    print("Try holding the 'ctrl' key to move a single vertex.")

    plt.show()

    selector.disconnect()

    # After figure is closed print the coordinates of the selected points
    print('\nSelected points:')
    print(selector.xys[selector.ind])
