'''
convert sc(shapep completion) dataset to npz file
'''
from __future__ import print_function
import numpy as np
from path import Path
import argparse
import pypcd
import os # listdir
import cnn3d.utils.grid_manip as gm

def write(records, fname):
    num = len(records)
    sx, sy, sz = records[0][0].shape
    g_x = np.empty((num, sx, sy, sz), dtype=np.bool)
    g_y = np.empty((num, sx, sy, sz), dtype=np.bool)

    for i in range(num):
        g_x[i], g_y[i] = records[i]
    np.savez_compressed(fname, g_x=g_x, g_y=g_y)

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=Path)
parser.add_argument('--obj-names', nargs='+', default='')
parser.add_argument('--dataset-name', type=Path, default='sc.npz')
parser.add_argument('--convert32', type=int, default=1)
parser.add_argument('--viz', type=int, default=0)
args = parser.parse_args()

records = []

sc_grid_size = (40, 40, 40) # original size

def pc2grid(pc, sz):
    g = np.zeros(sz, dtype=np.bool)
    for x, y, z in zip(pc.pc_data['x'].astype(np.uint8), pc.pc_data['y'].astype(np.uint8), pc.pc_data['z'].astype(np.uint8)):
        g[x, y, z] = True
    return g

import cnn3d.viz as viz
import IPython as ip

def convert2grid32(g):
    g = gm.pad_grid(g, (64, 64, 64))
    g = gm.rotate_grid(g, 180, -45, 0) # mimic our baxter's view point
    g = gm.shrink_half_grid(g)
    return g

if len(args.obj_names) == 0:
    # load all objects in 'data_dir'
    obj_names = os.listdir(args.data_dir)
else:
    obj_names = args.obj_names

for obj_name in obj_names:
    base_dir = (args.data_dir/obj_name).expand()

    for fname_x, fname_y in zip(sorted(base_dir.walkfiles('*_x.pcd')), sorted(base_dir.walkfiles('*_y.pcd'))):
        assert fname_x[:-6] == fname_y[:-6]
        # fname_x: partial point cloud (scanned)
        # fname_y: whole point cloud
        print(fname_x, fname_y)
        pc_x = pypcd.PointCloud.from_path(fname_x)
        pc_y = pypcd.PointCloud.from_path(fname_y)

        g_x = pc2grid(pc_x, sc_grid_size)
        g_y = pc2grid(pc_y, sc_grid_size)

        if args.convert32:
            g_x = convert2grid32(g_x)
            g_y = convert2grid32(g_y)

        if args.viz:
            viz.show_grid(g_x).show()
            viz.show_grid(g_y).show()
            ip.embed()

        records.append((g_x, g_y))

write(records, args.dataset_name)
