import time
from subprocess import call
import numpy as np
import math
import random
import vrep
from keras.models import load_model
from segmentation import segmentation


def remove_clipping(xyz):
    index = []
    for pts in range(0, len(xyz)):
        # calculate x index
        x = xyz[pts][0]
        y = xyz[pts][1]
        z = xyz[pts][2]
        # 0,-0.39098,0.13889
        if calculate_distance(x, y, z, 0, -0.5910, 0.7389) > 1.5:
            index.append(pts)
    xyz = np.delete(xyz, index, axis=0)
    return xyz


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
    viewpoint = "VIEWPOINT 0 0 0 0 1 0 0\n"
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


def setup_sim_camera(cid):
    '''
    Fetch pcd data and object matrix from V-RAP and transform to world frame
    :return: Raw pcd data
    '''
    # Get handle to camera
    sim_ret, cam_handle = vrep.simxGetObjectHandle(cid, 'kinect_depth', vrep.simx_opmode_blocking)
    emptyBuff = bytearray()
    res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'kinect',
                                                                                 vrep.sim_scripttype_childscript,
                                                                                 'absposition', [], [], [], emptyBuff,
                                                                                 vrep.simx_opmode_blocking)
    R = np.asarray([[retFloats[0], retFloats[1], retFloats[2], retFloats[3]],
                    [retFloats[4], retFloats[5], retFloats[6], retFloats[7]],
                    [retFloats[8], retFloats[9], retFloats[10], retFloats[11]]])
    print('camera pose is: ',R)
    result, state, data = vrep.simxReadVisionSensor(cid, cam_handle, vrep.simx_opmode_blocking)
    data = data[1]
    pcl = []
    for i in range(2, len(data), 4):
        p = [data[i], data[i + 1], data[i + 2], 1]
        pcl.append(np.matmul(R, p))
    return pcl


class GraspPoseGeneration(object):
    def __init__(self, pcd_path):
        self.path = pcd_path
        self.bottom = []
        self.surface = []
        self.axis = []
        self.approach = []
        self.binormal = []
        self.alpha = []
        self.beta = []
        self.gamma = []
        self.n_samples = 0
        self.rotm = []

    def generate_candidates(self):
        generate_candidates = '/home/lou00015/cnn3d/gpg/build/generate_candidates'
        config = '/home/lou00015/cnn3d/gpg/cfg/params.cfg'
        call([generate_candidates, config, self.path])
        time.sleep(5)
        f = open('candidates', 'r')
        data = f.readlines()
        nb_grasps = len(data)
        # candidates << vectorToString(hands[i].getGraspBottom()) << vectorToString(hands[i].getGraspSurface())
        #   << vectorToString(hands[i].getAxis()) << vectorToString(hands[i].getApproach())
        #   << vectorToString(hands[i].getBinormal()) << boost::lexical_cast<double>(hands[i].getGraspWidth()) << "\n";
        for i in range(nb_grasps):
            tmp_str = str.split(data[i], ',')
            tmp_float = [float(j) for j in tmp_str]
            tmp_apr = tmp_float[9:12]
            angle = np.dot(tmp_apr, [0, 0, 1])
            if angle < -0.2:
                self.bottom.append(tmp_float[0:3])
                self.surface.append(tmp_float[3:6])
                self.axis.append(tmp_float[6:9])
                self.approach.append(tmp_float[9:12])
                self.binormal.append(tmp_float[12:15])
                rotm = np.asarray([[tmp_float[12], tmp_float[6], tmp_float[9]],
                                   [tmp_float[13], tmp_float[7], tmp_float[10]],
                                   [tmp_float[14], tmp_float[8], tmp_float[11]]])
                self.rotm.append(rotm)
        self.n_samples = len(self.surface)


def add_three_objects(cid):
    object_name_list = []
    object_handle_list = []
    object_number = 3
    object_list = ['object_0','object_1','object_2','object_3','object_4','object_5']
    for i in range(object_number):
        object_name = random.choice(object_list)
        object_list.remove(object_name)
        object_name_list.append(object_name)
        print('Adding %s'%object_name)
        res, object_handle = vrep.simxGetObjectHandle(cid, object_name, vrep.simx_opmode_oneshot_wait)
        object_handle_list.append(object_handle)
        object_pos = [0,0,0.5]
        a = random.uniform(-90, 90)
        b = random.uniform(-90, 90)
        g = random.uniform(-90, 90)
        object_angle = [a,b,g]
        vrep.simxSetObjectPosition(cid,object_handle,-1,object_pos,vrep.simx_opmode_oneshot_wait)
        vrep.simxSetObjectOrientation(cid,object_handle,-1,object_angle,vrep.simx_opmode_oneshot_wait)
    return object_name_list, object_handle_list


def transform(pointcloud):
    voxel_grid = np.zeros((32, 32, 32), dtype=int)
    VOXEL_SIZE = 0.3/32
    for i in range(0, len(pointcloud)):
        x = 0
        y = 0
        z = 0
        for x_n in range(32):
            vg_min = -0.15+x_n*VOXEL_SIZE
            vg_max = vg_min+VOXEL_SIZE
            if vg_min<pointcloud[i][0]<vg_max:
                x = x_n
        for y_n in range(32):
            vg_min = -0.15+y_n*VOXEL_SIZE
            vg_max = vg_min+VOXEL_SIZE
            if vg_min<pointcloud[i][1]<vg_max:
                y = y_n
        for z_n in range(32):
            vg_min = z_n*VOXEL_SIZE
            vg_max = vg_min+VOXEL_SIZE
            if vg_min<pointcloud[i][2]<vg_max:
                z = z_n
        voxel_grid[x][y][z] = 1
    vg = voxel_grid.reshape((1,1,32,32,32))
    return vg


def push_scores(model, pointcloud):
    vg = transform(pointcloud)
    p = model.predict(vg, verbose=False)
    scores = p[0]
    return scores


def grasp_scores(model,candidates,pcl):
    score = []
    for i in range(candidates.n_samples):
        tmp = np.reshape(candidates.surface[i], (3,1))
        T_g2w = np.concatenate((candidates.rotm[i], tmp), axis=1)
        d = np.array([0,0,0,1])
        d = d.reshape((1,4))
        T_g2w = np.concatenate((T_g2w,d),axis=0)
        inv = np.linalg.inv(T_g2w)
        T_w2g = inv[:3, :]
        # print(T_w2g)
        data = np.zeros((len(pcl), 3))
        ones = np.ones((len(pcl), 1))
        pcl_tmp = np.append(pcl, ones, 1)
        for i in range(len(pcl)):
            data[i] = np.matmul(T_w2g, pcl_tmp[i])
        vg = transform(data)
        vg = vg.reshape((1,1,32,32,32))
        p = model.predict(vg, verbose=False)
        p = p[0]
        print(p)
        score.append(p)
    return score

if __name__ == '__main__':
    grasping_model = load_model('trained_models/grasping.h5')
    pushing_model = load_model('trained_models/pushing.h5')
    for layer in grasping_model.layers:
        layer.name = layer.name + '_grasping'
    for layer in pushing_model.layers:
        layer.name = layer.name + '_pushing'
    vrep.simxFinish(-1)
    # Connect to V-REP on port 19997
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if clientID != -1:
        print('Connected to simulation.')
        vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
        add_three_objects(clientID)
        pointcloud = setup_sim_camera(clientID)
        # Object segmentation
        pcd = remove_clipping(pointcloud)
        # v = pptk.viewer(pcd)
        # Save pcd file
        np.savetxt('data.pcd', pcd, delimiter=' ')
        insertHeader('data.pcd')
        # ----------------------------------------------------------------------------------------------------------
        # Push or grasp?
        # ----------------------------------------------------------------------------------------------------------
        # Push metrics
        push_sc = push_scores(pushing_model, pcd)
        print('Pushing metrics: ', push_sc)
        push_dir = np.where(push_sc == max(push_sc))

        # Grasp metrics
        nb_clutters = segmentation('data.pcd')
        print('Found %d objects' % nb_clutters)
        poses = GraspPoseGeneration('cloud_cluster_0.pcd')
        poses.generate_candidates()
        grasp_sc = grasp_scores(grasping_model, poses, pcd)
        print('Grasping metrics: ', grasp_sc)
        grasp_i = np.where(grasp_sc == max(grasp_sc))
        # Evaluate success rate of each action
        if poses.n_samples != 0 and max(push_sc) < max(grasp_sc):
            grasp(poses.rotm[grasp_i[0]], poses.surface[grasp_i[0]], clientID)
        else:
            push(push_dir, pcd, clientID)
            continue
    else:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
        exit()
