import time
from subprocess import call
import numpy as np
import math
import random
import vrep
from voxelization import transform
from keras.models import load_model
from segmentation import segmentation
import pptk

def voxel_write(index, poses, pcl):
    '''

    :return: voxel grid of the object based on current grasping pose
    '''
    # TODO: Transform pcd coordinates, voxelization and save voxel grid
    a = poses.surface[index]
    b = np.reshape(a,(3,1))
    c = poses.rotm[index]
    d = np.asarray([0, 0, 0, 1])
    d = np.reshape(d,(1,4))
    e = np.concatenate((c,b),axis=1)
    T_g2w = np.concatenate((e,d),axis=0)
    inv = np.linalg.inv(T_g2w)
    T_w2g = inv[:3, :]
    print(T_w2g)
    data = np.zeros((len(pcl), 3))
    ones = np.ones((len(pcl), 1))
    pcl_tmp = np.append(pcl, ones, 1)
    # print(np.shape(pcl_tmp))
    for i in range(len(pcl)):
        data[i] = np.matmul(T_w2g, pcl_tmp[i])
    # Convert to voxel grid
    vg = transform(data)
    return vg


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


def best_of(model,candidates,pcl):
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
    max_value = max(score)
    max_index = score.index(max_value)
    return max_value, max_index


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


def push():
    x_p = [[-0.125,0,0.05],[0.125,0,0.05]]
    x_n = [[0.125,0,0.05],[-0.125,0,0.05]]
    y_p = [[0,-0.125,0.05],[0,0.125,0.05]]
    y_n = [[0,0.125,0.05],[0,-0.125,0.05]]
    xy_p = [[-0.0884,-0.0884,0.05],[0.0884,0.0884,0.05]]
    xy_n = [[0.0884,0.0884,0.05],[-0.0884,-0.0884,0.05]]
    xp_yn = [[-0.0884,0.0884,0.05],[0.0884,-0.0884,0.05]]
    xn_yp = [[0.0884,-0.0884,0.05],[-0.0884,0.0884,0.05]]
    directions = [x_p,x_n,y_p,y_n,xy_p,xp_yn,xn_yp,xy_n]
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
            vg = transform(pcl)
            input_shape = vg.reshape((1,1,32,32,32))
            p = model.predict(input_shape, verbose=False)
            p = p[0]
            print('Predicted 8-dir success rate: ', p)
            best_dir = np.argmax(p)
            direction = directions[best_dir]
            print('The best direction is %d' % best_dir)
            f = open('predictions.txt', "a+")
            f.write(str(best_dir))
            f.close()
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
                print('execution signal sent')
                running = True
                while running:
                    res, signal = vrep.simxGetIntegerSignal(clientID, 'finish', vrep.simx_opmode_oneshot_wait)
                    if signal == tries:
                        running = False
                    else:
                        running = True
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




def single_object_evaluation():
    model = load_model('/tmp/cnn3d_train/grasping_epoch150_train_cnn3d_original.h5')
    experiment_number = 0
    vrep.simxFinish(-1)
    # Connect to V-REP on port 19997
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if clientID != -1:
        print('Connected to simulation.')
        while 1:
            vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
            time.sleep(5.0)
            vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
            emptyBuff = bytearray()
            # Generate object pcd file from V-REP
            # object_name, object_handle = add_one_object(clientID)
            res, object_handle, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, 'Sphere',
                                                                    vrep.sim_scripttype_childscript,
                                                                    'addRandObject', [], [], [],
                                                                    emptyBuff,
                                                                    vrep.simx_opmode_blocking)
            object_handle = object_handle[0]
            time.sleep(1.5)
            res, object_pos = vrep.simxGetObjectPosition(clientID,object_handle,-1,vrep.simx_opmode_oneshot_wait)
            pointcloud = setup_sim_camera(clientID)
            # Object segmentation
            pcd = remove_clipping(pointcloud)
            # v = pptk.viewer(pcd)
            # Save pcd file
            np.savetxt('data.pcd', pcd, delimiter=' ')
            insertHeader('data.pcd')
            # Generate grasp poses
            poses = GraspPoseGeneration('data.pcd')
            poses.generate_candidates()
            print('Number of valid grasps: ', poses.n_samples)
            # use learned 3dcnn to pick the best grasp
            if poses.n_samples != 0:
                value, index = best_of(model,poses,pcd)
            else:
                continue

            if value < 0.9:
                continue
            # --------------------------------------------------------------------------------------------------------------
            # execute
            # Get target and lift handles
            res, target1 = vrep.simxGetObjectHandle(clientID, 'grasp', vrep.simx_opmode_oneshot_wait)
            res, target2 = vrep.simxGetObjectHandle(clientID, 'lift', vrep.simx_opmode_oneshot_wait)

            # Set grasp position and orientation
            tmp_angles = rotationMatrixToEulerAngles(poses.rotm[index])
            angles = [-tmp_angles[0],-tmp_angles[1],-tmp_angles[2]]
            res1 = vrep.simxSetObjectPosition(clientID, target1, -1, poses.surface[index], vrep.simx_opmode_oneshot)
            res2 = vrep.simxSetObjectOrientation(clientID, target1, -1, angles, vrep.simx_opmode_oneshot)
            # Set lift position and orientation
            res3 = vrep.simxSetObjectPosition(clientID, target2, -1,
                                              [poses.surface[index][0], poses.surface[index][1], poses.surface[index][2] + 0.1],
                                              vrep.simx_opmode_oneshot)
            res4 = vrep.simxSetObjectOrientation(clientID, target2, -1, angles, vrep.simx_opmode_oneshot)
            time.sleep(1.0)
            print('Original Position is: ', object_pos)

            # Execute movements
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, 'Sphere',
                                                                                         vrep.sim_scripttype_childscript,
                                                                                         'go', [], [], [],
                                                                                         emptyBuff,
                                                                                         vrep.simx_opmode_blocking)
            running = True
            while running:
                res, signal = vrep.simxGetIntegerSignal(clientID, "finish", vrep.simx_opmode_oneshot_wait)
                if signal == 18:
                    running = False
                else:
                    running = True
            print('recording data ...')
            # Recording data
            res,current_pos = vrep.simxGetObjectPosition(clientID,object_handle,-1,vrep.simx_opmode_oneshot_wait)
            print('Current Position is: ', current_pos)
            if current_pos[2]>object_pos[2]+0.03:
                test_res = 1
            else:
                test_res = 0
            print(test_res)
            vg = voxel_write(index, poses, pcd)
            experiment_result = 'data/experiment_number_' + str(experiment_number)
            np.save(experiment_result, vg)
            experiment_number = experiment_number + 1
            f = open('label.txt', "a+")
            f.write(str(test_res))
            f.close()
            print('test completed, starting next iteration ...')
    else:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
        exit()


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


if __name__ == '__main__':
    # single_object_evaluation()
    # multiple_objects_evaluation()
    single_object_evaluation()
