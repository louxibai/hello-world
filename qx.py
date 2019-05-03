import time
import numpy as np
import math
import random
import vrep
from keras.models import load_model
from segmentation import segmentation
from transforms3d.euler import mat2euler
from utils import insertHeader, calculate_distance
from sampling import push_pose_generation, grasp_pose_generation, push_transform
import pypcd
import pptk
from shutil import copyfile


def rotate_cloud(point, pose, pcl):
    point = np.reshape(point,(3,1))
    dummy = np.asarray([0, 0, 0, 1])
    dummy = np.reshape(dummy,(1,4))
    T = np.concatenate((pose,point),axis=1)
    T_g2w = np.concatenate((T,dummy),axis=0)
    inv = np.linalg.inv(T_g2w)
    T_w2g = inv[:3, :]
    data = np.zeros((len(pcl), 3))
    ones = np.ones((len(pcl), 1))
    pcl_tmp = np.append(pcl, ones, 1)
    # print(np.shape(pcl_tmp))
    for i in range(len(pcl)):
        data[i] = np.matmul(T_w2g, pcl_tmp[i])
    return data


def voxelize(pointcloud, scale):
    voxel_grid = np.zeros((32, 32, 32), dtype=int)
    VOXEL_SIZE = scale/32
    for i in range(0, len(pointcloud)):
        x = 0
        y = 0
        z = 0
        for x_n in range(32):
            vg_min = -scale/2+x_n*VOXEL_SIZE
            vg_max = vg_min+VOXEL_SIZE
            if vg_min<pointcloud[i][0]<vg_max:
                x = x_n
        for y_n in range(32):
            vg_min = -scale/2+y_n*VOXEL_SIZE
            vg_max = vg_min+VOXEL_SIZE
            if vg_min<pointcloud[i][1]<vg_max:
                y = y_n
        for z_n in range(32):
            vg_min = -scale/2+z_n*VOXEL_SIZE
            vg_max = vg_min+VOXEL_SIZE
            if vg_min<pointcloud[i][2]<vg_max:
                z = z_n
        voxel_grid[x][y][z] = 1
    vg = voxel_grid.reshape((1,1,32,32,32))
    return vg


def add_one_object(cid):
    object_index = random.randint(0,5)
    object_name = 'object_'+str(object_index)
    print('Adding object_%d'%object_index)
    res, object_handle = vrep.simxGetObjectHandle(cid, object_name, vrep.simx_opmode_oneshot_wait)
    object_pos = [0,0,0.3]
    a = random.uniform(-90, 90)
    b = random.uniform(-90, 90)
    g = random.uniform(-90, 90)
    object_angle = [a,b,g]
    vrep.simxSetObjectPosition(cid,object_handle,-1,object_pos,vrep.simx_opmode_oneshot_wait)
    vrep.simxSetObjectOrientation(cid,object_handle,-1,object_angle,vrep.simx_opmode_oneshot_wait)
    return object_name, object_handle


def remove_clipping(xyz):
    index = []
    for pts in range(0, len(xyz)):
        # calculate x index
        x = xyz[pts][0]
        y = xyz[pts][1]
        z = xyz[pts][2]
        # 0,-0.39098,0.13889
        if calculate_distance(x, y, z, 0, -0.5910, 0.7389) > 1.5 or x > 0.5 or y > 0.5:
            index.append(pts)
    xyz = np.delete(xyz, index, axis=0)
    return xyz



def add_multiple_objects(cid, nb_obj):
    object_name_list = []
    object_handle_list = []
    object_number = nb_obj
    # object_list = ['imported_part_0','imported_part_1','imported_part_2','imported_part_3','imported_part_4','imported_part_5','imported_part_6','imported_part_7']
    object_list = ['imported_part_0','imported_part_1','imported_part_2','imported_part_3','imported_part_4','imported_part_5','imported_part_6','imported_part_7']
    # object_list = ['imported_part_0','imported_part_6','imported_part_7']
    for i in range(object_number):
        object_name = random.choice(object_list)
        object_list.remove(object_name)
        object_name_list.append(object_name)
        print('Adding %s'%object_name)
        res, object_handle = vrep.simxGetObjectHandle(cid, object_name, vrep.simx_opmode_oneshot_wait)
        object_handle_list.append(object_handle)
        object_pos = [0,0,0.3]
        a = random.uniform(-90, 90)
        b = random.uniform(-90, 90)
        g = random.uniform(-90, 90)
        object_angle = [a,b,g]
        vrep.simxSetObjectPosition(cid,object_handle,-1,object_pos,vrep.simx_opmode_oneshot_wait)
        vrep.simxSetObjectOrientation(cid,object_handle,-1,object_angle,vrep.simx_opmode_oneshot_wait)
    return object_name_list, object_handle_list


def push_scores(model, push_poses, pointcloud):
    scores = []
    for i in range(len(push_poses)):
        pose = push_poses[i]
        push_point = pose[0]
        push_angle = pose[1]
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
        data = np.zeros((len(pointcloud), 3))
        ones = np.ones((len(pointcloud), 1))
        pcl_tmp = np.append(pointcloud, ones, 1)
        for i in range(len(pointcloud)):
            data[i] = np.matmul(T_w2g, pcl_tmp[i])
        vg = voxelize(data, 0.4)
        p = model.predict(vg, verbose=False)
        p = p[0]
        scores.append(p)
    return scores


def grasp_scores(model, grasp_poses, surface, pointcloud):
    score = []
    for i in range(len(surface)):
        tmp = np.reshape(surface[i], (3,1))
        T_g2w = np.concatenate((grasp_poses[i], tmp), axis=1)
        d = np.array([0,0,0,1])
        d = d.reshape((1,4))
        T_g2w = np.concatenate((T_g2w,d),axis=0)
        inv = np.linalg.inv(T_g2w)
        T_w2g = inv[:3, :]
        data = np.zeros((len(pointcloud), 3))
        ones = np.ones((len(pointcloud), 1))
        pcl_tmp = np.append(pointcloud, ones, 1)
        for i in range(len(pointcloud)):
            data[i] = np.matmul(T_w2g, pcl_tmp[i])
        vg = voxelize(data, 0.2)
        vg = vg.reshape((1,1,32,32,32))
        p = model.predict(vg, verbose=False)
        p = p[0]
        score.append(p)
    return score


class Panda(object):
    def __init__(self, clientID):
        self.cid = clientID
        self.dummybyte = bytearray()

    def init_env(self):
        panda_id = self.cid
        vrep.simxStopSimulation(panda_id, vrep.simx_opmode_oneshot_wait)
        time.sleep(5.0)
        vrep.simxStartSimulation(panda_id, vrep.simx_opmode_oneshot_wait)
        names, handles = add_multiple_objects(panda_id,8)
        object_pos = []
        object_ori = []
        time.sleep(5)
        for obj_i in range(8):
            res, pos = vrep.simxGetObjectPosition(panda_id,handles[obj_i],-1,vrep.simx_opmode_oneshot_wait)
            res, ori = vrep.simxGetObjectOrientation(panda_id,handles[obj_i],-1,vrep.simx_opmode_oneshot_wait)
            object_pos.append(pos)
            object_ori.append(ori)
        return object_pos, object_ori, handles

    def get_cloud(self):
        panda_id = self.cid
        # Get handle to camera
        sim_ret, cam_handle = vrep.simxGetObjectHandle(panda_id, 'kinect_depth', vrep.simx_opmode_blocking)
        # Get camera pose and intrinsics in simulation
        emptyBuff = self.dummybyte
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(panda_id, 'kinect',
                                                                                     vrep.sim_scripttype_childscript,
                                                                                     'absposition', [], [], [], emptyBuff,
                                                                                     vrep.simx_opmode_blocking)
        R = np.asarray([[retFloats[0], retFloats[1], retFloats[2], retFloats[3]],
                        [retFloats[4], retFloats[5], retFloats[6], retFloats[7]],
                        [retFloats[8], retFloats[9], retFloats[10], retFloats[11]]])
        # print('camera pose is: ',R)
        result, state, data = vrep.simxReadVisionSensor(panda_id, cam_handle, vrep.simx_opmode_blocking)
        data = data[1]
        pcl = []
        for i in range(2, len(data), 4):
            p = [data[i], data[i + 1], data[i + 2], 1]
            pcl.append(np.matmul(R, p))
        pcd = remove_clipping(pcl)
        np.savetxt('data.pcd', pcd, delimiter=' ')
        insertHeader('data.pcd')
        return pcd

    def grasp(self, grasp_pose, surface):
        # Set up grasping position and orientation
        emptyBuff = self.dummybyte
        panda_id = self.cid
        # Get target and lift handles
        res, target1 = vrep.simxGetObjectHandle(panda_id, 'grasp', vrep.simx_opmode_oneshot_wait)
        res, target2 = vrep.simxGetObjectHandle(panda_id, 'lift', vrep.simx_opmode_oneshot_wait)

        # Set grasp position and orientation
        tmp_angles = mat2euler(grasp_pose)
        angles = [-tmp_angles[0],-tmp_angles[1],-tmp_angles[2]]
        res1 = vrep.simxSetObjectPosition(panda_id, target1, -1, surface, vrep.simx_opmode_oneshot)
        res2 = vrep.simxSetObjectOrientation(panda_id, target1, -1, angles, vrep.simx_opmode_oneshot)
        # Set lift position and orientation
        res3 = vrep.simxSetObjectPosition(panda_id, target2, -1,
                                          [surface[0], surface[1], surface[2] + 0.1],
                                          vrep.simx_opmode_oneshot)
        res4 = vrep.simxSetObjectOrientation(panda_id, target2, -1, angles, vrep.simx_opmode_oneshot)
        time.sleep(1.0)

        # --------------------------------------------------------------------------------------------------------------
        # Execute movements
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(panda_id, 'Sphere',
                                                                                     vrep.sim_scripttype_childscript,
                                                                                     'grasp', [], [], [],
                                                                                     emptyBuff,
                                                                                     vrep.simx_opmode_blocking)
        running = True
        while running:
            res, signal = vrep.simxGetIntegerSignal(panda_id, "finish", vrep.simx_opmode_oneshot_wait)
            if signal == 18:
                running = False
            else:
                running = True

    def push(self, push_pose):
        emptyBuff = self.dummybyte
        panda_id = self.cid
        # ----------------------------------------------------------------------------------------------------------------------
        # Push Pose Generation
        res, target1 = vrep.simxGetObjectHandle(panda_id, 'lift0', vrep.simx_opmode_oneshot_wait)
        res, target2 = vrep.simxGetObjectHandle(panda_id, 'grasp', vrep.simx_opmode_oneshot_wait)
        res, target3 = vrep.simxGetObjectHandle(panda_id, 'lift', vrep.simx_opmode_oneshot_wait)
        angles = push_pose[1]
        # Set pushing point and orientation
        res1 = vrep.simxSetObjectPosition(panda_id, target1, -1, [push_pose[0][0],push_pose[0][1],push_pose[0][2]+0.25], vrep.simx_opmode_oneshot)
        res2 = vrep.simxSetObjectOrientation(panda_id, target1, -1, angles, vrep.simx_opmode_oneshot)
        # Set landing position
        res1 = vrep.simxSetObjectPosition(panda_id, target2, -1, push_pose[2], vrep.simx_opmode_oneshot)
        res2 = vrep.simxSetObjectOrientation(panda_id, target2, -1, angles, vrep.simx_opmode_oneshot)
        # Set pushing direction
        res3 = vrep.simxSetObjectPosition(panda_id, target3, -1, push_pose[3], vrep.simx_opmode_oneshot)
        res4 = vrep.simxSetObjectOrientation(panda_id, target3, -1, angles, vrep.simx_opmode_oneshot)
        time.sleep(10)
        # Execute movements
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(panda_id, 'Sphere',
                                                                                     vrep.sim_scripttype_childscript,
                                                                                     'push', [], [], [],
                                                                                     emptyBuff,
                                                                                     vrep.simx_opmode_blocking)
        print('pushing signal sent')
        running = True
        while running:
            res, signal = vrep.simxGetIntegerSignal(panda_id, "finish", vrep.simx_opmode_oneshot_wait)
            if signal == 18:
                running = False
            else:
                running = True


def multiple_objects_evaluation_no_pushing():
    grasping_model = load_model('trained_models/grasping.h5')
    # for layer in grasping_model.layers:
    #     layer.name = layer.name + '_grasping'
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if clientID != -1:
        while True:
            # Initialize environment
            panda = Panda(clientID)
            obj_pos, obj_ori, handles = panda.init_env()
            handles_left = len(handles)
            while handles_left>0:
                pointcloud = panda.get_cloud()
                grasp_poses, points = grasp_pose_generation('data.pcd')
                grasp_sc = grasp_scores(grasping_model, grasp_poses, points, pointcloud)
                print('Grasping scores:')
                for scores in grasp_sc:
                    print(scores)
                # ----------------------------------------------------------------------------------------------------------
                if len(grasp_sc) != 0:
                    max_index = np.argmax(grasp_sc)
                    print('Highest grasping score index: ', max_index)
                    panda.grasp(grasp_poses[max_index], points[max_index])
                # ----------------------------------------------------------------------------------------------------------
                print('Checking results')
                for j in range(3):
                    res,current_pos = vrep.simxGetObjectPosition(clientID,handles[j],-1,vrep.simx_opmode_oneshot_wait)
                    print(current_pos)
                    if current_pos[2]>obj_pos[j][2]+0.03:
                        vrep.simxSetObjectPosition(clientID,handles[j],-1, [2,2,0.5], vrep.simx_opmode_oneshot_wait)
                        obj_pos[j] = [2,2,0.5]
                        handles_left = handles_left-1
                vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
                time.sleep(3)
                vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
                for k in range(3):
                    vrep.simxSetObjectPosition(clientID,handles[k],-1,obj_pos[k],vrep.simx_opmode_blocking)
                    vrep.simxSetObjectOrientation(clientID,handles[k],-1,obj_ori[k],vrep.simx_opmode_blocking)
    else:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
        exit()


def grasping_data():
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    test=15379
    if clientID != -1:
        while True:
            # Initialize environment
            panda = Panda(clientID)
            obj_pos, obj_ori, handles = panda.init_env()
            pointcloud = panda.get_cloud()
            grasp_poses, points = grasp_pose_generation('data.pcd')
            # ----------------------------------------------------------------------------------------------------------
            for g_index in range(len(grasp_poses)):
                print(grasp_poses[g_index],points[g_index])
                panda.grasp(grasp_poses[g_index], points[g_index])
                res,current_pos = vrep.simxGetObjectPosition(clientID,handles[0],-1,vrep.simx_opmode_oneshot_wait)
                print(current_pos)
                if current_pos[2]>obj_pos[0][2]+0.03:
                    result=1
                else:
                    result=0
                r_cloud = rotate_cloud(points[g_index],grasp_poses[g_index],pointcloud)
                np.savetxt('forthehorde.pcd', r_cloud, fmt='%1.9f', delimiter=' ')
                insertHeader('forthehorde.pcd')
                copyfile('/home/lou00015/cnn3d/scripts/forthehorde.pcd','/home/lou00015/dataset/data_UR5/test'+str(test)+'.pcd')
                f = open('/home/lou00015/dataset/data_UR5/label.txt', "a+")
                f.write(str(result))
                f.close()
                test = test+1
                vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
                time.sleep(3)
                vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
    else:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
    exit()


def pushing_data():
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    test=1336
    if clientID != -1:
        while True:
            # Initialize environment
            panda = Panda(clientID)
            obj_pos, obj_ori, handles = panda.init_env()
            pointcloud = panda.get_cloud()
            nb_clutters = segmentation('data.pcd')
            print("Found %d clouds before pushing" % nb_clutters)
            push_poses = push_pose_generation(pointcloud, 30)
            # ----------------------------------------------------------------------------------------------------------
            for p_index in range(len(push_poses)):
                panda.push(push_poses[p_index])
                cloud_new = panda.get_cloud()
                nb_clutters_new = segmentation('data.pcd')
                print("Found %d clouds after pushing" % nb_clutters_new)
                if nb_clutters_new>nb_clutters:
                    result=1
                else:
                    result=0
                r_cloud = push_transform(push_poses[p_index],pointcloud)
                np.savetxt('forthehorde.pcd', r_cloud, fmt='%1.9f', delimiter=' ')
                insertHeader('forthehorde.pcd')
                copyfile('/home/lou00015/cnn3d/scripts/forthehorde.pcd','/home/lou00015/dataset/push_UR5/test'+str(test)+'.pcd')
                f = open('/home/lou00015/dataset/push_UR5/label.txt', "a+")
                f.write(str(result))
                f.close()
                test = test+1
                vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
                time.sleep(3)
                vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
                for j in range(8):
                    vrep.simxSetObjectPosition(clientID,handles[j],-1,obj_pos[j],vrep.simx_opmode_oneshot_wait)
                    vrep.simxSetObjectOrientation(clientID,handles[j],-1,obj_ori[j],vrep.simx_opmode_oneshot_wait)
    else:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
    exit()


def debugging():
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    grasps = ['hand0', 'hand1', 'hand2', 'hand3', 'hand4', 'hand5', 'hand6']
    if clientID != -1:
        # Initialize environment
        panda = Panda(clientID)
        panda.init_env()
        # while 1:
        pointcloud = panda.get_cloud()
        grasp_poses, points = grasp_pose_generation('data.pcd')
        for i in range(len(grasp_poses)):
            grasp_pose = grasp_poses[i]
            surface = points[i]
            matrix = [grasp_pose[0][0],grasp_pose[0][1],grasp_pose[0][2],surface[0],grasp_pose[1][0],grasp_pose[1][1],
                      grasp_pose[1][2],surface[1],grasp_pose[2][0],grasp_pose[2][1],grasp_pose[2][2],surface[2]]
            emptyBuff = bytearray()
            panda_id = clientID
            grasp = grasps[i]
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(panda_id, grasp,
                                                                                     vrep.sim_scripttype_childscript,
                                                                                     'setmatrix', [], matrix, [], emptyBuff,
                                                                                     vrep.simx_opmode_blocking)
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(panda_id, grasp,
                                                                                         vrep.sim_scripttype_childscript,
                                                                                         'absposition', [], [], [], emptyBuff,
                                                                                         vrep.simx_opmode_blocking)
            R = np.asarray([[retFloats[0], retFloats[1], retFloats[2], retFloats[3]],
                            [retFloats[4], retFloats[5], retFloats[6], retFloats[7]],
                            [retFloats[8], retFloats[9], retFloats[10], retFloats[11]]])
            print(R)

    else:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
        exit()


if __name__ == '__main__':
    # multiple_objects_evaluation_no_pushing()
    # more_data()
    pushing_data()
    # debugging()
