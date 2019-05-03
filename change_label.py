#!/usr/bin/env python

'''
print '[side] 0: ->, 1: /, 2: |, 3: \\, 4: <-'
print '[top] 5: --, 6: /, 7: |, 8, \\'
'''
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import os
from path import Path
import copy
from cnn3d import viz

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=Path)
parser.add_argument('obj_names', nargs='+')
args = parser.parse_args()

for obj_name in args.obj_names:
    base_dir = (args.data_dir/obj_name).expand()

    for fname in sorted(base_dir.walkfiles('*.npz')):
        data = np.load(fname)
        g = data['g']
        gt = list(data['gt'])
        gt_old = copy.copy(gt)

        print fname
        print gt

        print 'annotate grasping'
        print '[side] 0: ->, 1: /, 2: |, 3: \\, 4: <-'
        print '[top] 5: --, 6: /, 7: |, 8, \\'
        print 'type q to quit annotation'
        f = viz.show_grid(g)
        f.show()

        while True:
            key_a = raw_input()
            if key_a is 'q' or key_a is 'g':
                break

            if key_a == '' or not key_a.isdigit() or int(key_a) < 0 or int(key_a) > 8:
                print 'unkown value, try again'
            else:

                print 'key pressed: %d' % int(key_a)
                if int(key_a) in gt:
                    print int(key_a), 'is in gt, will remove it'
                    gt.remove(int(key_a))
                else:
                    print int(key_a), 'is inserted'
                    gt.append(int(key_a))
                print 'new gt: ', gt

        if set(gt) != set(gt_old): # it gt is changed
            print 'want to save? (y/n)'
            key = raw_input()
            if key == 'y':
                np.savez_compressed(fname, g=g, gt=sorted(gt))

