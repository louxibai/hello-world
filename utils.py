import numpy as np
import os
import math
from keras.models import load_model
from shutil import copyfile
from voxelization import transform
import pypcd
import pptk


def insertHeader(filename):
    num_lines = sum(1 for line in open(filename))
    title = "# .PCD v0.7 - Point Cloud Data file format\n"
    version = "VERSION 0.7\n"
    fields = "FIELDS x y z\n"
    size = "SIZE 4 4 4\n"
    type = "TYPE F F F\n"
    count = "COUNT 1 1 1\n"
    width = "WIDTH " + str(num_lines) + "\n"
    height = "HEIGHT 1\n"
    viewpoint = "VIEWPOINT 0 0 0 1 0 0 0\n"
    points = "POINTS " + str(num_lines) + "\n"
    d_type = "DATA ascii\n"
    pcd_header = [title, version, fields, size, type, count, width, height, viewpoint, points, d_type]
    f = open(filename, "r")
    contents = f.readlines()
    f.close()
    for i in range(len(pcd_header)):
        contents.insert(i, pcd_header[i])
    f = open(filename, "w")
    f.writelines(contents)
    f.close()


def calculate_distance(x1, y1, z1, x2, y2, z2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return dist


def find_fails(directory):
    f = open('label.txt','r')
    data = f.readlines()
    data = data[0]
    fails = []
    l = []
    for i in range(len(data)):
        result = int(data[i])
        if result == 0:
            failed = np.load(directory+'/experiment_number_'+str(i)+'.npy')
            for j in range(9):
                fails.append(failed)
                l.append(0)
    np.savez('grasping_train_20181220.npz', g = fails, gt = l)


def calculate_succ(exp_nb):
    l=[]
    f = open('/home/lou00015/dataset/data_final/label.txt')
    label = f.readlines()
    label = label[0]
    for i in range(exp_nb+1):
        l.append(int(label[i]))
    total = np.sum(l, axis=0)
    rate = np.true_divide(total,exp_nb)
    return rate


def generate_npz(directory):
    # move_label()
    count = 0
    for name in os.listdir(directory+'/data'):
        count = count+1
    print(count)

    vg = []
    l = []
    for i in range(1273):
        data = np.load(directory+'/data/experiment_number_'+str(i)+'.npy')
        label = np.load(directory+'/pushing_label/label_'+str(i)+'.npy')
        vg.append(data)
        l.append(label)
    np.savez('/home/lou00015/cnn3d/dataset/pushing/train.npz', g = vg, gt = l)

    vg_t = []
    l_t = []
    # for j in range(int(math.floor(0.8*count)),count-1):
    for j in range(1273,1274):
        data = np.load(directory+'/data/experiment_number_'+str(j)+'.npy')
        label = np.load(directory+'/pushing_label/label_'+str(j)+'.npy')
        vg_t.append(data)
        l_t.append(label)
    np.savez('/home/lou00015/cnn3d/dataset/pushing/test.npz', g = vg_t, gt = l_t)


def merge_train_npz():
    # path1 = raw_input('npz file 1:')
    # path2 = raw_input('npz file 2:')
    path1 = '/home/lou00015/cnn3d/dataset/pushing/train_14k.npz'
    path2 = '/home/lou00015/cnn3d/dataset/pushing/train_15k_patch.npz'
    # load 1st dataset
    data1 = np.load(path1)
    vg1 = data1['g']
    print(np.shape(vg1))
    l1 = data1['gt']
    # load 2nd dataset
    data2 = np.load(path2)
    vg2 = data2['g']
    print(np.shape(vg2))
    l2 = data2['gt']
    # concatenate two dataset
    vg = np.concatenate((vg1,vg2), axis=0)
    l = np.concatenate((l1,l2), axis=0)
    print(np.shape(vg))
    print(np.shape(l))
    np.savez_compressed('/home/lou00015/cnn3d/dataset/pushing/train_15k.npz', g = vg, gt = l)
    print("Training data created")


def merge_test_npz():
    # path1 = raw_input('npz file 1:')
    # path2 = raw_input('npz file 2:')
    path1 = '/home/lou00015/cnn3d/dataset/grasping/test_0123.npz'
    path2 = '/home/lou00015/cnn3d/dataset/grasping/grasping_test_data.npz'
    # load 1st dataset
    data1 = np.load(path1)
    vg1 = data1['g']
    print(np.shape(vg1))
    l1 = data1['gt']
    # load 2nd dataset
    data2 = np.load(path2)
    vg2 = data2['g']
    print(np.shape(vg2))
    l2 = data2['gt']
    # concatenate two dataset
    vg = np.concatenate((vg1,vg2), axis=0)
    l = np.concatenate((l1,l2), axis=0)
    np.savez('/home/lou00015/cnn3d/dataset/grasping/grasping_test_data_tmp.npz', g = vg, gt = l)
    os.remove("/home/lou00015/cnn3d/dataset/grasping/grasping_test_data.npz")
    os.rename('/home/lou00015/cnn3d/dataset/grasping/grasping_test_data_tmp.npz','/home/lou00015/cnn3d/dataset/grasping/grasping_test_data.npz')
    os.remove("/home/lou00015/cnn3d/dataset/grasping/grasping_test_data_tmp.npz")
    print("Testing data created")


def _load_npz(fn):
    f = np.load(fn)
    x = f['g']
    y = f['gt']
    x = x.reshape(x.shape[0], 1, 32, 32, 32)
    x = x.astype('float32')
    y = y.astype('float32')
    return x, y


def load(train_fname):
    resolution = 32
    x_train, y_train = _load_npz(train_fname)
    orig_resolution = x_train.shape[-1]
    div = int(orig_resolution/resolution)
    x_train = x_train[:, :, ::div, ::div, ::div]
    input_shape = (1, resolution, resolution, resolution)
    num_class = 1
    return x_train, y_train


def train(x_train, y_train):
    model = load_model('/tmp/cnn3d_train/grasping_epoch150_train_cnn3d_original.h5')
    history = model.fit(x_train, y_train, batch_size=32, epochs=150,
                        verbose=1, validation_split=0.1)
    # score = model.evaluate(x_test, y_test, verbose=0)
    model.save('20190111.h5')
    return history


def voxelize():
    directory = '/home/lou00015/dataset/push_UR5'
    label_path = '/home/lou00015/dataset/push_UR5/label.txt'
    f = open(label_path,'r')
    tmp = f.readlines()
    label = tmp[0]

    g = []
    gt = []
    for tests in range(1300):
        filename = directory+'/test'+str(tests)+'.pcd'
        vg = transform(filename)
        print('Voxel grid %d created...' % tests)
        g.append(vg)
        gt.append(label[tests])
    np.savez_compressed('/home/lou00015/cnn3d/dataset/grasping/pushing_UR5.npz', g = g, gt = gt)


def pose_save():
    directory = '/home/lou00015/dataset/collision_nn'
    g = []
    for tests in range(11000):
        fn = directory+'/pose_'+str(tests)+'.npy'
        print(fn)
        pose = np.load(fn)
        a = pose.tolist()
        g.append(a)
    np.savez_compressed('/home/lou00015/cnn3d/dataset/cf/input_pose.npz', g = g)


def rename():
    directory = '/home/lou00015/dataset/panda_grasping_pcd'
    i = 37749
    j = 0
    for filename in os.listdir(directory):
        dst = '/home/lou00015/dataset/panda_pcd/experiment_number_' + str(i) + ".pcd"
        src = directory + '/experiment_number_' + str(j) + '.pcd'
        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1
        j += 1


def viewcloud():
    pc = pypcd.PointCloud.from_path('cloud_cluster_0.pcd')
    data = [pc.pc_data['x'],pc.pc_data['y'],pc.pc_data['z']]
    pointcloud = np.transpose(data)

    v = pptk.viewer(pointcloud)


if __name__ == '__main__':
    # pose_save()
    voxelize()
    # print(calculate_succ(4400))
