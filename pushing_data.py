import time
import numpy as np
import math
import random
import vrep
from shutil import copyfile
import pptk
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
    # Get camera pose and intrinsics in simulation
    # a, b, g = angles[0], angles[1], angles[2]
    # Rx = np.array([[1.0, 0.0, 0.0],
    #                [0.0, np.cos(a), -np.sin(a)],
    #                [0.0, np.sin(a), np.cos(a)]], dtype=np.float32)
    # Ry = np.array([[np.cos(b), 0.0,  np.sin(b)],
    #                [0.0, 1.0, 0.0],
    #                [-np.sin(b), 0.0, np.cos(b)]], dtype=np.float32)
    # Rz = np.array([[np.cos(g), -np.sin(g), 0.0],
    #                [np.sin(g), np.cos(g), 0.0],
    #                [0.0, 0.0, 1.0]], dtype=np.float32)
    # R = np.dot(Rz, np.dot(Ry, Rx))
    emptyBuff = bytearray()
    res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'kinect',
                                                                                 vrep.sim_scripttype_childscript,
                                                                                 'absposition', [], [], [], emptyBuff,
                                                                                 vrep.simx_opmode_blocking)
    R = np.asarray([[retFloats[0], retFloats[1], retFloats[2], retFloats[3]],
                    [retFloats[4], retFloats[5], retFloats[6], retFloats[7]],
                    [retFloats[8], retFloats[9], retFloats[10], retFloats[11]]])
    result, state, data = vrep.simxReadVisionSensor(cid, cam_handle, vrep.simx_opmode_blocking)
    data = data[1]
    pcl = []
    for i in range(2, len(data), 4):
        p = [data[i], data[i + 1], data[i + 2], 1]
        pcl.append(np.matmul(R, p))
    return pcl


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return [x, y, z]


def pcd_transform(push_pose, pcl):
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
    print(T_w2g)
    data = np.zeros((len(pcl), 3))
    ones = np.ones((len(pcl), 1))
    pcl_tmp = np.append(pcl, ones, 1)
    for i in range(len(pcl)):
        data[i] = np.matmul(T_w2g, pcl_tmp[i])
    return data


def add_three_objects(cid):
    object_name_list = []
    object_handle_list = []
    object_number = 3
    object_list = ['imported_part_0','imported_part_1','imported_part_2','imported_part_3','imported_part_4','imported_part_5', 'imported_part_6', 'imported_part_7']
    for i in range(object_number):
        object_name = random.choice(object_list)
        object_list.remove(object_name)
        object_name_list.append(object_name)
        print('Adding %s'%object_name)
        res, object_handle = vrep.simxGetObjectHandle(cid, object_name, vrep.simx_opmode_oneshot_wait)
        object_handle_list.append(object_handle)
        object_pos = [0,0,0.25]
        a = random.uniform(-90, 90)
        b = random.uniform(-90, 90)
        g = random.uniform(-90, 90)
        object_angle = [a,b,g]
        vrep.simxSetObjectPosition(cid,object_handle,-1,object_pos,vrep.simx_opmode_oneshot_wait)
        vrep.simxSetObjectOrientation(cid,object_handle,-1,object_angle,vrep.simx_opmode_oneshot_wait)
    return object_name_list, object_handle_list


def push_pose_generation(pcd, nb):
    filter_list = []
    for i in range(len(pcd)):
        if pcd[i][2]<0.01:
            filter_list.append(i)
    filtered = np.delete(pcd, filter_list, axis=0)
    push_list = []
    for j in range(nb):
        pt = random.choice(filtered)
        gamma = float(random.randint(0,628))/100.0
        angle = [-3.14, 0, gamma]
        landing = [pt[0]-0.1*math.sin(gamma), pt[1]-0.1*math.cos(gamma), pt[2]]
        ending = [pt[0]+0.1*math.sin(gamma), pt[1]+0.1*math.cos(gamma), pt[2]]
        pose = [pt, angle, landing, ending]
        push_list.append(pose)
    return push_list


def push(push_pose, clientID):
    emptyBuff = bytearray()
    # ----------------------------------------------------------------------------------------------------------------------
    # Push Pose Generation
    res, target1 = vrep.simxGetObjectHandle(clientID, 'lift0', vrep.simx_opmode_oneshot_wait)
    res, target2 = vrep.simxGetObjectHandle(clientID, 'grasp', vrep.simx_opmode_oneshot_wait)
    res, target3 = vrep.simxGetObjectHandle(clientID, 'lift', vrep.simx_opmode_oneshot_wait)
    angles = push_pose[1]
    # Set pushing point and orientation
    res1 = vrep.simxSetObjectPosition(clientID, target1, -1, push_pose[0], vrep.simx_opmode_oneshot)
    res2 = vrep.simxSetObjectOrientation(clientID, target1, -1, angles, vrep.simx_opmode_oneshot)
    # Set landing position
    res1 = vrep.simxSetObjectPosition(clientID, target2, -1, push_pose[2], vrep.simx_opmode_oneshot)
    res2 = vrep.simxSetObjectOrientation(clientID, target2, -1, angles, vrep.simx_opmode_oneshot)
    # Set pushing direction
    res3 = vrep.simxSetObjectPosition(clientID, target3, -1, push_pose[3], vrep.simx_opmode_oneshot)
    res4 = vrep.simxSetObjectOrientation(clientID, target3, -1, angles, vrep.simx_opmode_oneshot)
    time.sleep(10)
    # Execute movements
    res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, 'Sphere',
                                                                                 vrep.sim_scripttype_childscript,
                                                                                 'push', [], [], [],
                                                                                 emptyBuff,
                                                                                 vrep.simx_opmode_blocking)
    print('pushing signal sent')
    running = True
    while running:
        res, signal = vrep.simxGetIntegerSignal(clientID, "finish", vrep.simx_opmode_oneshot_wait)
        if signal == 18:
            running = False
        else:
            running = True


def cnn3d_mp_data():
    experiment_number = 0
    vrep.simxFinish(-1)
    # Connect to V-REP on port 19997
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if clientID != -1:
        print('Connected to simulation.')
        while 1:
            # ----------------------------------------------------------------------------------------------------------
            vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
            # Generate object pcd file from V-REP
            names, handles = add_three_objects(clientID)
            num_obj = 3
            object_pos = []
            object_ori = []
            time.sleep(3)
            for obj_i in range(num_obj):
                res, pos = vrep.simxGetObjectPosition(clientID,handles[obj_i],-1,vrep.simx_opmode_oneshot_wait)
                res, ori = vrep.simxGetObjectOrientation(clientID,handles[obj_i],-1,vrep.simx_opmode_oneshot_wait)
                object_pos.append(pos)
                object_ori.append(ori)
            pointcloud = setup_sim_camera(clientID)
            # Object segmentation
            pcd = remove_clipping(pointcloud)
            np.savetxt('data.pcd', pcd, delimiter=' ')
            insertHeader('data.pcd')
            # ----------------------------------------------------------------------------------------------------------
            # Push
            nb_clutters = segmentation('data.pcd')
            if nb_clutters==num_obj:
                vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
                time.sleep(3)
                continue
            print('Found %d objects' % nb_clutters)
            pushes = push_pose_generation(pcd, 10)
            vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
            for j in range(num_obj):
                vrep.simxSetObjectPosition(clientID,handles[j],-1,object_pos[j],vrep.simx_opmode_oneshot_wait)
                vrep.simxSetObjectOrientation(clientID,handles[j],-1,object_ori[j],vrep.simx_opmode_oneshot_wait)
            push_pose = pushes[i]
            print('Point: ', push_pose[0], 'Angle: ', push_pose[1], 'From: ', push_pose[2], 'to: ' ,push_pose[3])
            push(push_pose, clientID)
            tb_save = pcd_transform(push_pose,pcd)
            np.savetxt('for_the_horde.pcd', tb_save, delimiter=' ')
            insertHeader('for_the_horde.pcd')
            experiment_result = 'experiment_number_' + str(experiment_number)+'.pcd'
            src = '/home/louxibai/Research/cnn3d/scripts/for_the_horde.pcd'
            dst = '/home/louxibai/Research/cnn3d_data/pcd/' + experiment_result
            copyfile(src, dst)
            experiment_number = experiment_number + 1
            new_raw_pcd = setup_sim_camera(clientID)
            vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
            # Object segmentation
            new_pcd = remove_clipping(new_raw_pcd)
            np.savetxt('new_data.pcd', new_pcd, delimiter=' ')
            insertHeader('new_data.pcd')
            nb_clutters_new = segmentation('new_data.pcd')
            print('Found %d objects' % nb_clutters_new)
            if nb_clutters_new>nb_clutters:
                test_result = 1
                print('Push success!')
            else:
                print('Push fail')
                test_result = 0
            f = open('/home/louxibai/Research/cnn3d_data/panda_label.txt', "a+")
            f.write(str(test_result))
            f.close()
            # ----------------------------------------------------------------------------------------------------------
            print('test completed, starting next iteration ...')
    else:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
        exit()


def pushing_data_collection():
    experiment_number = 0
    vrep.simxFinish(-1)
    # Connect to V-REP on port 19997
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if clientID != -1:
        print('Connected to simulation.')
        while 1:
            # ----------------------------------------------------------------------------------------------------------
            vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
            # Generate object pcd file from V-REP
            names, handles = add_three_objects(clientID)
            num_obj = 3
            object_pos = []
            object_ori = []
            time.sleep(3)
            for obj_i in range(num_obj):
                res, pos = vrep.simxGetObjectPosition(clientID,handles[obj_i],-1,vrep.simx_opmode_oneshot_wait)
                res, ori = vrep.simxGetObjectOrientation(clientID,handles[obj_i],-1,vrep.simx_opmode_oneshot_wait)
                object_pos.append(pos)
                object_ori.append(ori)
            pointcloud = setup_sim_camera(clientID)
            # Object segmentation
            pcd = remove_clipping(pointcloud)
            np.savetxt('data.pcd', pcd, delimiter=' ')
            insertHeader('data.pcd')
            # ----------------------------------------------------------------------------------------------------------
            # Push
            nb_clutters = segmentation('data.pcd')
            if nb_clutters==num_obj:
                vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
                time.sleep(3)
                continue
            print('Found %d objects' % nb_clutters)
            pushes = push_pose_generation(pcd, 10)
            for i in range(10):
                vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
                for j in range(num_obj):
                    vrep.simxSetObjectPosition(clientID,handles[j],-1,object_pos[j],vrep.simx_opmode_oneshot_wait)
                    vrep.simxSetObjectOrientation(clientID,handles[j],-1,object_ori[j],vrep.simx_opmode_oneshot_wait)
                push_pose = pushes[i]
                print('Point: ', push_pose[0], 'Angle: ', push_pose[1], 'From: ', push_pose[2], 'to: ' ,push_pose[3])
                push(push_pose, clientID)
                tb_save = pcd_transform(push_pose,pcd)
                np.savetxt('for_the_horde.pcd', tb_save, delimiter=' ')
                insertHeader('for_the_horde.pcd')
                experiment_result = 'experiment_number_' + str(experiment_number)+'.pcd'
                src = '/home/louxibai/Research/cnn3d/scripts/for_the_horde.pcd'
                dst = '/home/louxibai/Research/cnn3d_data/pcd/' + experiment_result
                copyfile(src, dst)
                experiment_number = experiment_number + 1
                new_raw_pcd = setup_sim_camera(clientID)
                vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
                # Object segmentation
                new_pcd = remove_clipping(new_raw_pcd)
                np.savetxt('new_data.pcd', new_pcd, delimiter=' ')
                insertHeader('new_data.pcd')
                nb_clutters_new = segmentation('new_data.pcd')
                print('Found %d objects' % nb_clutters_new)
                if nb_clutters_new>nb_clutters:
                    test_result = 1
                    print('Push success!')
                else:
                    print('Push fail')
                    test_result = 0
                f = open('/home/louxibai/Research/cnn3d_data/panda_label.txt', "a+")
                f.write(str(test_result))
                f.close()

            # ----------------------------------------------------------------------------------------------------------
            print('test completed, starting next iteration ...')
    else:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
        exit()


if __name__ == '__main__':
    cnn3d_mp_data()
