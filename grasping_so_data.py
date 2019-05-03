import time
import numpy as np
import math
import random
import vrep
from transforms3d.euler import mat2euler
from utils import insertHeader, calculate_distance
from sampling import push_pose_generation, grasp_pose_generation, new_grasp_pose_generation
import pypcd
import pptk
from shutil import copyfile


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
    object_list = ['imported_part_0','imported_part_1','imported_part_2','imported_part_3','imported_part_4','imported_part_5']
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


class Panda(object):
    def __init__(self, clientID):
        self.cid = clientID
        self.dummybyte = bytearray()

    def init_env(self):
        panda_id = self.cid
        vrep.simxStopSimulation(panda_id, vrep.simx_opmode_oneshot_wait)
        time.sleep(5.0)
        vrep.simxStartSimulation(panda_id, vrep.simx_opmode_oneshot_wait)
        names, handles = add_multiple_objects(panda_id,1)
        object_pos = []
        object_ori = []
        time.sleep(3)
        for obj_i in range(1):
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
        # Set grasp position and orientation
        matrix = [grasp_pose[0][0],grasp_pose[1][0],grasp_pose[2][0],surface[0],grasp_pose[0][1],grasp_pose[1][1],
                  grasp_pose[2][1],surface[1],grasp_pose[0][2],grasp_pose[1][2],grasp_pose[2][2],surface[2]]
        vrep.simxCallScriptFunction(panda_id, 'grasp', vrep.sim_scripttype_childscript, 'setmatrix', [], matrix, [], emptyBuff, vrep.simx_opmode_blocking)
        time.sleep(1.0)

        # --------------------------------------------------------------------------------------------------------------
        # Execute movements
        vrep.simxCallScriptFunction(panda_id, 'Sphere', vrep.sim_scripttype_childscript, 'grasp', [], [], [], emptyBuff, vrep.simx_opmode_blocking)
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
        # # After pushing
        # result, state, new_data = vrep.simxReadVisionSensor(panda_id, cam_handle, vrep.simx_opmode_blocking)
        # new_data = new_data[1]
        # new_pointcloud = []
        # for i in range(2, len(data), 4):
        #     p = [new_data[i], new_data[i + 1], new_data[i + 2], 1]
        #     new_pointcloud.append(np.matmul(R, p))
        #     # Object segmentation
        # new_pcl = remove_clipping(new_pointcloud)
        # np.savetxt('data_new.pcd', new_pcl, delimiter=' ')
        # insertHeader('data_new.pcd')
        # nb_clutters_new = segmentation('data_new.pcd')
        # print('Second try: %d objects segmented' % nb_clutters_new)
        # if nb_clutters_new>nb_clutters:
        #     test_result = 1
        #     print('Push success!')
        # else:
        #     print('Push fail')
        #     test_result = 0
        # f = open('label.txt', "a+")
        # f.write(str(test_result))
        # f.close()

    def reset(self):
        # Set up grasping position and orientation
        emptyBuff = self.dummybyte
        panda_id = self.cid
        # --------------------------------------------------------------------------------------------------------------
        # Execute movements
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(panda_id, 'Sphere',
                                                                                     vrep.sim_scripttype_childscript,
                                                                                     'reset', [], [], [],
                                                                                     emptyBuff,
                                                                                     vrep.simx_opmode_blocking)
        running = True
        while running:
            res, signal = vrep.simxGetIntegerSignal(panda_id, "finish", vrep.simx_opmode_oneshot_wait)
            if signal == 18:
                running = False
            else:
                running = True


def single_object_evaluation():
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    test=4824
    if clientID != -1:
        while True:
            # Initialize environment
            panda = Panda(clientID)
            obj_pos, obj_ori, handles = panda.init_env()
            pointcloud = panda.get_cloud()
            grasp_poses, points = new_grasp_pose_generation(pointcloud,10)
            print('Found %d poses' % len(grasp_poses))
            res, tip_hdl = vrep.simxGetObjectHandle(clientID,'tip',vrep.simx_opmode_blocking)
            res, tip_xyz = vrep.simxGetObjectPosition(clientID,tip_hdl,-1,vrep.simx_opmode_oneshot_wait)
            print('tip is at: ', tip_xyz)
            # ----------------------------------------------------------------------------------------------------------
            for i in range(len(grasp_poses)):
                panda.grasp(grasp_poses[i], points[i])
                filename = '/home/lou00015/dataset/collision_nn/pose_'+str(test)+'.npy'
                tbsaved = grasp_poses[i]
                tbsaved.append(points[i])
                np.save(filename,tbsaved)
                # ----------------------------------------------------------------------------------------------------------
                res,current_pos = vrep.simxGetObjectPosition(clientID,tip_hdl,-1,vrep.simx_opmode_oneshot_wait)
                print('current tip at: ', current_pos)
                if current_pos[0] != tip_xyz[0]:
                    label=1
                else:
                    label=0
                copyfile('/home/lou00015/cnn3d/scripts/data.pcd','/home/lou00015/dataset/collision_nn/test'+str(test)+'.pcd')
                f = open('/home/lou00015/dataset/collision_nn/label.txt', "a+")
                f.write(str(label))
                f.close()
                test = test+1
                vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
                time.sleep(3)
                vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
                vrep.simxSetObjectPosition(clientID,handles[0],-1,obj_pos[0],vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(clientID,handles[0],-1,obj_ori[0],vrep.simx_opmode_blocking)
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
        grasp_poses, points = new_grasp_pose_generation(pointcloud,6)
        for i in range(len(grasp_poses)):
            grasp_pose = grasp_poses[i]
            print(grasp_pose)
            surface = points[i]
            matrix = [grasp_pose[0][0],grasp_pose[1][0],grasp_pose[2][0],surface[0],grasp_pose[0][1],grasp_pose[1][1],
                      grasp_pose[2][1],surface[1],grasp_pose[0][2],grasp_pose[1][2],grasp_pose[2][2],surface[2]]
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
    # debugging()
    single_object_evaluation()
