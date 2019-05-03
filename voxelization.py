import numpy as np
import pypcd

def transform(filename):
    pc = pypcd.PointCloud.from_path(filename)
    data = [pc.pc_data['x'],pc.pc_data['y'],pc.pc_data['z']]
    pointcloud = np.transpose(data)
    # initialize 32x32x32 voxel grid
    voxel_grid = np.zeros((32, 32, 32), dtype=int)
    VOXEL_SIZE = 0.2/32

    for i in range(0, len(pointcloud)):
        x = 0
        y = 0
        z = 0
        for x_n in range(32):
            vg_min = -0.1+x_n*VOXEL_SIZE
            vg_max = vg_min+VOXEL_SIZE
            if vg_min<pointcloud[i][0]<vg_max:
                x = x_n
        for y_n in range(32):
            vg_min = -0.1+y_n*VOXEL_SIZE
            vg_max = vg_min+VOXEL_SIZE
            if vg_min<pointcloud[i][1]<vg_max:
                y = y_n
        for z_n in range(32):
            vg_min = -0.1+z_n*VOXEL_SIZE
            vg_max = vg_min+VOXEL_SIZE
            if vg_min<pointcloud[i][2]<vg_max:
                z = z_n
        voxel_grid[x][y][z] = 1
    return voxel_grid


def test():
    pc = pypcd.PointCloud.from_path('/home/lou00015/dataset/pcd/experiment_number_1.pcd')
    data = np.array([pc.pc_data['x'],pc.pc_data['y'],pc.pc_data['z']])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(data, edgecolor='k')

    plt.show()


import pptk

def viewcloud():
    pc = pypcd.PointCloud.from_path('/home/lou00015/dataset/pcd/experiment_number_1.pcd')
    data = [pc.pc_data['x'],pc.pc_data['y'],pc.pc_data['z']]
    pointcloud = np.transpose(data)

    v = pptk.viewer(pointcloud)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(filename):
    data = np.load(filename)
    print(np.shape(data))
    pcd = data['g']
    label = data['gt']
    # prepare some coordinates
    print(label[0:20])
    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(pcd[500], edgecolor='k')

    plt.show()


if __name__ == '__main__':
    # filename = '/home/lou00015/cnn3d/scripts/pcd/experiment_number_1500.pcd'
    # pc = pypcd.PointCloud.from_path('/home/lou00015/dataset/pcd/experiment_number_2.pcd')
    # data = [pc.pc_data['x'],pc.pc_data['y'],pc.pc_data['z']]
    # center = np.mean(data, axis=1)
    # print(center)
    # data = np.transpose(data)
    # vg = transform(filename)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.voxels(vg, edgecolor='k')
    # plt.show()
    # viewcloud()
    visualize('/home/lou00015/cnn3d/dataset/grasping/train_20cm_13000.npz')

