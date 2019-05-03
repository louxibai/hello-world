
'''
convert grasping dataset to npz file
'''

from __future__ import print_function

import numpy as np
from path import Path
import argparse

numof_dir = 9

def write(records, fname):
    num = len(records)
    sx, sy, sz = records[0][0].shape
    nd = records[0][1].shape[0]
    gg = np.empty((num, sx, sy, sz), dtype=np.float32)
    gt = np.empty((num, nd), dtype=np.float32)

    for i in range(num):
        gg[i], gt[i] = records[i]
    np.savez_compressed(fname, g=gg, gt=gt)

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=Path)
parser.add_argument('obj_names', nargs='+')
parser.add_argument('--dataset-name', type=Path, default='grasping_train.npz')
parser.add_argument('--mode', type=int, default=1)
parser.add_argument('--center', type=bool, default=0)
args = parser.parse_args()

records = []

def center_grid(g):
    if g.dtype == 'bool':
        x, y, z = np.nonzero(g)
    else:
        x, y, z = np.nonzero(g+1.0)

    sx, sy, sz = g.shape
    mx, my, mz = (np.min(x)+np.max(x))/2.0, (np.min(y)+np.max(y))/2.0, (np.min(z)+np.max(z))/2.0
    shift_x = int((sx-1.0)/2.0 - mx)
    shift_y = int((sy-1.0)/2.0 - my)
    shift_z = int((sz-1.0)/2.0 - mz)
    
    # g = np.roll(g, shift_x, axis=0)
    # g = np.roll(g, shift_y, axis=1)
    g = np.roll(g, shift_z, axis=2)
    return g

for obj_name in args.obj_names:
    base_dir = (args.data_dir/obj_name).expand()

    for fname in sorted(base_dir.walkfiles('*.npz')):
        data = np.load(fname)
        g = data['g']
        gt = data['gt']

        if args.center:
            # center voxel grid
            g = center_grid(g)

        if args.mode == 0: # only one label for each object
            for gt_ in gt:
                records.append((g, gt_))
        elif args.mode == 1: # multiple labels
            y = np.zeros(numof_dir, dtype=np.float32)
            y[gt] = 1
            records.append((g, y))
        print(fname)
        print(gt)

write(records, args.dataset_name)
