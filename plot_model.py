from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import IPython as ip

plt.style.use('seaborn-paper')

from keras.models import load_model
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib as mpl

import argparse

def cuboid_data(pos, size=(1,1,1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return x, y, z

def plotCubeAt(pos=(0,0,0), ax=None, alpha=1.0, color='darkorange', size=(1,1,1), zorder=1, linewidth=0):
    # Plotting a cube element at position pos
    if ax != None:
        X, Y, Z = cuboid_data(pos, size)
        ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=alpha, shade=True, edgecolor=color, zorder=zorder, linewidth=linewidth)

# def plotBoundary(pos=(0,0,0), ax=None, alpha=1.0, color='darkorange', size=(1,1,1)):
#     if ax != None:
#         pos = list(pos)
#         size = list(size)
#         x = [pos[0], pos[0]+size[0], pos[0]+size[0], pos[0], pos[0]]
#         y = [pos[1], pos[1], pos[1]+size[1], pos[1]+size[1], pos[1]]
#         z = [pos[2]+size[2], pos[2]+size[2], pos[2]+size[2], pos[2]+size[2], pos[2]+size[2]]
#         ax.plot(x, y, z, color=color, alpha=alpha, zorder=1000)

class ModelViz():
    def __init__(self, fn):
        self._fn = fn
        self._model = load_model(self._fn)

        self._w = []
        for l in self._model.layers:
            # self._weights = layer.get_weights()
            if l.name[:6] == 'conv3d':
                self._w.append(l.get_weights())

    # def get_filters(layer_idx=0):
    def viz_filters(self, layer_idx):
        w = self._w[layer_idx][0] # [1] is bias
        sx, sy, sz, _, N = w.shape # N: num of filters
        cx, cy, cz = (sx-1)/2, (sy-1)/2, (sz-1)/2 # center indices

        plt.figure(figsize=(16,2))
        for i in range(N):
            f_ = w[:, :, :, 0, i]
            # slices
            f_x = f_[cx, :, :]
            f_y = f_[:, cy, :]
            f_z = f_[:, :, cz]

            plt.subplot(3, N, N*0 + i + 1)
            plt.imshow(f_x, cmap=plt.cm.gray, interpolation='nearest')
            plt.axis('off')

            plt.subplot(3, N, N*1 + i + 1)
            plt.imshow(f_y, cmap=plt.cm.gray, interpolation='nearest')
            plt.axis('off')

            plt.subplot(3, N, N*2 + i + 1)
            plt.imshow(f_z, cmap=plt.cm.gray, interpolation='nearest')
            plt.axis('off')
        # plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.0)
        # plt.colorbar()
        plt.show()

    def _viz_cubes(self, ax, f, norm, slice_x=False, slice_y=False, slice_z=False):
        ax.set_aspect('equal')
        sx, sy, sz = f.shape
        rx, ry, rz = range(sx), range(sy), range(sz)

        if slice_x:
            rx = rx[(sx-1)/2:]
        if slice_y:
            ry = ry[:(sy+1)/2]
        if slice_z:
            rz = rz[:(sz+1)/2]


        for x_ in rx:
            for y_ in ry:
                for z_ in rz:
                    plotCubeAt(pos=(x_, y_, z_), ax=ax, color=cm.gray(norm(f[x_, y_, z_])), alpha=1.0)
        # plotCubeAt(pos=(sx/2, sy/2, sz/2), ax=ax, color='#ff7f0e', alpha=0.0, size=tuple(f.shape), zorder=0, linewidth=0.1)

        ax.set_xlim3d(0, sx)
        ax.set_ylim3d(0, sy)
        ax.set_zlim3d(0, sz)
        # fix the rotation
        ax.view_init(30, 180-40)
        ax.grid(False)
        ax.axis('off')


    def viz_filters_3d(self, layer_idx):
        w = self._w[layer_idx][0] # [1] is bias

        w_max = np.max(w)
        w_min = np.min(w)
        norm = colors.Normalize(w_min, w_max)
        sx, sy, sz, _, N = w.shape # N: num of filters
        cx, cy, cz = (sx-1)/2, (sy-1)/2, (sz-1)/2 # center indices

        N = 16
        fig = plt.figure(figsize=(N,2))
        for i in range(N):
            f = w[:, :, :, 0, i]
            ax = fig.add_subplot(3, N, N*0 + i + 1, projection='3d')
            self._viz_cubes(ax, f, norm, slice_x=True)

            ax = fig.add_subplot(3, N, N*1 + i + 1, projection='3d')
            self._viz_cubes(ax, f, norm, slice_y=True)

            ax = fig.add_subplot(3, N, N*2 + i + 1, projection='3d')
            self._viz_cubes(ax, f, norm, slice_z=True)

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', '--file_name', type=str, required=True, help='cnn3d model file name (h5)')
    args = parser.parse_args()

    mv = ModelViz(args.file_name)
    # mv.viz_filters(0)
    mv.viz_filters_3d(0)
    mv.viz_filters_3d(1)

