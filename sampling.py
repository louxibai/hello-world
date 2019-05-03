import math
import random
import numpy as np
import time
import subprocess


def push_pose_generation(pcd, nb):
    push_list = []
    for j in range(nb):
        pt = random.choice(pcd)
        pt = [pt[0],pt[1],pt[2]+0.03]
        gamma = float(random.randint(0,628))/100.0
        angle = [-3.14, 0, gamma]
        landing = [pt[0]-0.1*math.sin(gamma), pt[1]-0.1*math.cos(gamma), pt[2]]
        ending = [pt[0]+0.1*math.sin(gamma), pt[1]+0.1*math.cos(gamma), pt[2]]
        pose = [pt, angle, landing, ending]
        push_list.append(pose)
    return push_list

def push_transform(push_pose, pcl):
    push_point = push_pose[0]
    push_angle = push_pose[1]
    a,b,g = push_angle[0],push_angle[1],push_angle[2]
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(a), -np.sin(a)],
                   [0.0, np.sin(a), np.cos(a)]], dtype=np.float32)
    Ry = np.array([[np.cos(b), 0.0,  np.sin(b)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(b), 0.0, np.cos(b)]], dtype=np.float32)
    Rz = np.array([[np.cos(g), -np.sin(g), 0.0],
                   [np.sin(g), np.cos(g), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    R = np.dot(Rz, np.dot(Ry, Rx))
    translation = np.reshape(push_point,(3,1))
    dummy = np.asarray([0, 0, 0, 1])
    h_g = np.reshape(dummy,(1,4))
    R_upper = np.concatenate((R,translation),axis=1)
    T_g2w = np.concatenate((R_upper,h_g),axis=0)
    inv = np.linalg.inv(T_g2w)
    T_w2g = inv[:3, :]
    data = np.zeros((len(pcl), 3))
    ones = np.ones((len(pcl), 1))
    pcl_tmp = np.append(pcl, ones, 1)
    for i in range(len(pcl)):
        data[i] = np.matmul(T_w2g, pcl_tmp[i])
    return data

def grasp_pose_generation(filename):
    poses = []
    bottom = []
    path = filename
    # FNULL = open(os.devnull, 'w')
    generate_candidates = '/home/lou00015/cnn3d/gpg/build/generate_candidates'
    config = '/home/lou00015/cnn3d/gpg/cfg/params.cfg'
    subprocess.call([generate_candidates, config, path])
    # subprocess.call(generate_candidates)
    time.sleep(5)
    f = open('candidates', 'r')
    data = f.readlines()
    nb_grasps = len(data)
    for i in range(nb_grasps):
        tmp_str = str.split(data[i], ',')
        tmp_float = [float(j) for j in tmp_str]
        tmp_apr = tmp_float[9:12]
        angle = np.dot(tmp_apr, [0, 0, 1])
        if angle < -0.2:
            bottom.append(tmp_float[0:3])
            rotm = np.asarray([[tmp_float[12], tmp_float[6], tmp_float[9]],
                               [tmp_float[13], tmp_float[7], tmp_float[10]],
                               [tmp_float[14], tmp_float[8], tmp_float[11]]])
            poses.append(rotm)
    filtered = len(bottom)
    print('%d grasps after filtering' % filtered)
    return poses, bottom


def generate_rand_pose():
    ap_x = random.uniform(-100,100)
    ap_y = random.uniform(-100,100)
    ap_z = random.uniform(-100,0)
    ap_d = math.sqrt(ap_x*ap_x+ap_y*ap_y+ap_z*ap_z)
    approach = ap_x, ap_y, ap_z
    bi_x = random.uniform(-100,100)
    bi_y = random.uniform(-100,100)
    bi_z = (0-ap_x*bi_x-ap_y*bi_y)/ap_z
    binormal = bi_x, bi_y, bi_z
    bi_d = math.sqrt(bi_x*bi_x+bi_y*bi_y+bi_z*bi_z)
    norm_x = np.true_divide(binormal,bi_d)
    norm_x = np.reshape(norm_x,(3,1))
    norm_y = np.cross(approach, binormal)
    d = math.sqrt(np.square(norm_y[0])+np.square(norm_y[1])+np.square(norm_y[2]))
    norm_y = np.true_divide(norm_y,d)
    norm_y = np.reshape(norm_y,(3,1))
    norm_z = np.true_divide(approach,ap_d)
    norm_z = np.reshape(norm_z,(3,1))
    norm = np.concatenate((norm_x,norm_y,norm_z), axis=1)
    return norm


def new_grasp_pose_generation(pcd,nb):
    pose = []
    point = []
    for j in range(nb):
        pt = random.choice(pcd)
        point.append(pt)
        xyz = generate_rand_pose()
        xyz = xyz.tolist()
        pose.append(xyz)
    return pose, point


if __name__ == '__main__':
    data = np.load('/home/lou00015/dataset/collision_nn/pose_0.npy')
    print(data)
