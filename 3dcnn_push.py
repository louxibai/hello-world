import time
import numpy as np
import math
import vrep
# from keras.models import load_model
from voxelization import transform
from segmentation import segmentation
import pptk
import random


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
        object_pos = [0,0,0.25]
        a = random.uniform(-90, 90)
        b = random.uniform(-90, 90)
        g = random.uniform(-90, 90)
        object_angle = [a,b,g]
        vrep.simxSetObjectPosition(cid,object_handle,-1,object_pos,vrep.simx_opmode_oneshot_wait)
        vrep.simxSetObjectOrientation(cid,object_handle,-1,object_angle,vrep.simx_opmode_oneshot_wait)
    return object_name_list, object_handle_list


from shutil import copyfile


if __name__ == '__main__':
    # model = load_model('/home/lou00015/cnn3d/scripts/trained_models/pushing_8_dir.h5')
    experiment_number = 18369
    height = 0.03
    x_p = [[-0.125,0,height],[0.125,0,height]]
    x_n = [[0.125,0,height],[-0.125,0,height]]
    y_p = [[0,-0.125,height],[0,0.125,height]]
    y_n = [[0,0.125,height],[0,-0.125,height]]
    xy_p = [[-0.0884,-0.0884,height],[0.0884,0.0884,height]]
    xy_n = [[0.0884,0.0884,height],[-0.0884,-0.0884,height]]
    xp_yn = [[-0.0884,0.0884,height],[0.0884,-0.0884,height]]
    xn_yp = [[0.0884,-0.0884,height],[-0.0884,0.0884,height]]
    directions = [x_p,x_n,y_p,y_n,xy_p,xp_yn,xn_yp,xy_n]
    vrep.simxFinish(-1)
    # Connect to V-REP on port 19997
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if clientID != -1:
        print('Connected to simulation.')
        vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
        while 1:
            vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
            time.sleep(3.0)
            vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
            # add object
            object_name, object_handle = add_three_objects(clientID)
            time.sleep(5.0)
            num_obj = 3
            object_pos = []
            object_angle = []
            for obj_i in range(num_obj):
                res, pos = vrep.simxGetObjectPosition(clientID,object_handle[obj_i],-1,vrep.simx_opmode_oneshot_wait)
                res, orientation = vrep.simxGetObjectOrientation(clientID,object_handle[obj_i],-1,vrep.simx_opmode_oneshot_wait)
                object_pos.append(pos)
                object_angle.append(orientation)
            # Generate object pcd file from V-REP
            sim_ret, cam_handle = vrep.simxGetObjectHandle(clientID, 'kinect_depth', vrep.simx_opmode_blocking)
            emptyBuff = bytearray()
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, 'kinect',
                                                                                         vrep.sim_scripttype_childscript,
                                                                                         'absposition', [], [], [], emptyBuff,
                                                                                         vrep.simx_opmode_blocking)
            R = np.asarray([[retFloats[0], retFloats[1], retFloats[2], retFloats[3]],
                            [retFloats[4], retFloats[5], retFloats[6], retFloats[7]],
                            [retFloats[8], retFloats[9], retFloats[10], retFloats[11]]])
            # print('camera pose is: ',R)
            result, state, data = vrep.simxReadVisionSensor(clientID, cam_handle, vrep.simx_opmode_blocking)
            data = data[1]
            pointcloud = []
            for i in range(2, len(data), 4):
                p = [data[i], data[i + 1], data[i + 2], 1]
                pointcloud.append(np.matmul(R, p))
                # Object segmentation
            pcl = remove_clipping(pointcloud)
            # Save pcd file
            np.savetxt('data.pcd', pcl, delimiter=' ')
            insertHeader('data.pcd')
            nb_clutters = segmentation('data.pcd')
            label = [0,0,0,0,0,0,0,0]
            if nb_clutters == num_obj:
                continue
            print('Number of objects: %d' % nb_clutters)
            tries = 1
            for direction in directions:
                res, target1 = vrep.simxGetObjectHandle(clientID, 'grasp', vrep.simx_opmode_oneshot_wait)
                res, target2 = vrep.simxGetObjectHandle(clientID, 'lift', vrep.simx_opmode_oneshot_wait)
                res, target3 = vrep.simxGetObjectHandle(clientID, 'lift0', vrep.simx_opmode_oneshot_wait)

                angles = [-3.14, 0, 0]
                # Set landing position
                res1 = vrep.simxSetObjectPosition(clientID, target1, -1, direction[0], vrep.simx_opmode_oneshot)
                res2 = vrep.simxSetObjectOrientation(clientID, target1, -1, angles, vrep.simx_opmode_oneshot)
                # Set pushing direction
                res3 = vrep.simxSetObjectPosition(clientID, target2, -1, direction[1], vrep.simx_opmode_oneshot)
                res4 = vrep.simxSetObjectOrientation(clientID, target2, -1, angles, vrep.simx_opmode_oneshot)
                # Set wait position
                res5 = vrep.simxSetObjectPosition(clientID, target3, -1, [direction[1][0],direction[1][1],direction[1][2]+0.15], vrep.simx_opmode_oneshot)
                res6 = vrep.simxSetObjectOrientation(clientID, target3, -1, angles, vrep.simx_opmode_oneshot)
                # Execute movements
                res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, 'Sphere',
                                                                                             vrep.sim_scripttype_childscript,
                                                                                             'go', [], [], [],
                                                                                             emptyBuff,
                                                                                             vrep.simx_opmode_blocking)
                running = True
                fault = 0
                while running:
                    res, signal = vrep.simxGetIntegerSignal(clientID, 'finish', vrep.simx_opmode_oneshot_wait)
                    res, fault = vrep.simxGetIntegerSignal(clientID, 'fault', vrep.simx_opmode_oneshot_wait)
                    if fault == 1:
                        running = False
                    if signal == tries:
                        running = False
                    else:
                        running = True
                if fault == 1:
                    print('system fault, stopping ...')
                    break
                print('recording data ...')
                # Recording data
                time.sleep(1.0)
                # After pushing
                result, state, new_data = vrep.simxReadVisionSensor(clientID, cam_handle, vrep.simx_opmode_blocking)
                new_data = new_data[1]
                new_pointcloud = []
                for i in range(2, len(data), 4):
                    p = [new_data[i], new_data[i + 1], new_data[i + 2], 1]
                    new_pointcloud.append(np.matmul(R, p))
                    # Object segmentation
                new_pcl = remove_clipping(new_pointcloud)
                np.savetxt('data_new.pcd', new_pcl, delimiter=' ')
                insertHeader('data_new.pcd')
                # v = pptk.viewer(new_pcl) # Visualize pcd if needed
                nb_clutters_new = segmentation('data_new.pcd')
                print('Number of objects: %d' % nb_clutters_new)
                if nb_clutters_new>nb_clutters:
                    dir_index = directions.index(direction)
                    print('Tried direction:', direction)
                    print('Number %d in directions list' % dir_index)
                    label[dir_index]=1
                    print('Updated label:', label)
                else:
                    print('Pushing not meaningful ...')
                for j in range(num_obj):
                    vrep.simxSetObjectPosition(clientID, object_handle[j], -1, object_pos[j], vrep.simx_opmode_oneshot_wait)
                    vrep.simxSetObjectOrientation(clientID, object_handle[j], -1, object_angle[j], vrep.simx_opmode_oneshot_wait)
                time.sleep(0.5)
                tries = tries+1
            print(label)
            if fault == 1:
                print('fault skipped, data not recorded.')
                continue
            experiment_result = 'pcd/experiment_number_'+str(experiment_number)
            pushing_label = '/home/lou00015/dataset/pushing_label_panda/label_'+str(experiment_number)
            src = '/home/lou00015/cnn3d/scripts/data.pcd'
            dst = '/home/lou00015/dataset/'+experiment_result+'.pcd'
            copyfile(src, dst)
            np.save(pushing_label,label)
            experiment_number = experiment_number+1
            print('Execution completed, starting next iteration ...')
    else:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
        exit()
