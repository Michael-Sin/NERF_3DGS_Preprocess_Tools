import cv2
import os
import numpy as np 
import colmap_read_model
from plyfile import PlyData
import numpy as np
from scipy.spatial.transform import Rotation as R


def check_path(dir_path):
    isExist = os.path.exists(dir_path)
    if not isExist:
        os.makedirs(dir_path)


def create_transform_matrix(tvec, qvec, qw):
    #x, y, z, qx, qy, qz, qw = data
    x, y, z = tvec
    qx, qy, qz = qvec
    rotation_matrix = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    transform_matrix = np.eye(4)
    transform_matrix[0:3, 0:3] = rotation_matrix
    transform_matrix[0:3, 3] = [x, y, z]
    transform_matrix[3, 3] = 1
    return transform_matrix
 

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    rotation_matrix = np.array([[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                               [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                               [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]])
    return rotation_matrix


def rotation_matrix_to_quaternion(transform_matrix):
    r = R.from_matrix(transform_matrix)
    q_scalar_last = r.as_quat()
    q_scalar_first = np.array([q_scalar_last[3], q_scalar_last[0], q_scalar_last[1], q_scalar_last[2]])
    return q_scalar_first


def motion_blur(ori_path, direction, kernel_size):
    dir_path = ori_path+"_"+"blurred"+"_"+direction+"_"+str(kernel_size)
    check_path(dir_path)

    # Motion Blur
    for filename in os.listdir(ori_path):
        img = cv2.imread(ori_path+"/"+filename)
        kernel = np.zeros((kernel_size, kernel_size)) 
        if direction == "horizontal":
            kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
        elif direction == "vertical":
            kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
        kernel /= kernel_size 

        motion_blur_img = cv2.filter2D(img, -1, kernel) 
        cv2.imwrite(dir_path+"/"+filename, motion_blur_img)


def vid2imgs(vid_path, img_path):

    # Read vid and convert to imgs
    cam = cv2.VideoCapture(vid_path) 
    currentframe = 1
    counter = 0
    empty_name = "000000"
    while(True): 
        ret,frame = cam.read()
        if counter % 1 == 0:
            if ret: 
                name = img_path + "/" + empty_name[len(str(currentframe)):] + str(currentframe) + '.jpg'
                cv2.imwrite(name, frame) 
                currentframe += 1
            else: 
                break
        counter += 1
    cam.release() 
    cv2.destroyAllWindows()


def downsample(ori_path, down_factor):
    isExist = os.path.exists(ori_path)
    if not isExist:
        print("ERROR: No such directory, please revise your dataset.")
    down_sample_path = ori_path+"_"+str(down_factor)
    check_path(down_sample_path)

    # Downsample images
    for filename in os.listdir(ori_path):
        image = cv2.imread(ori_path+"/"+filename)
        new_width = int(image.shape[1]/down_factor)
        new_height = int(image.shape[0]/down_factor)
        img_half = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(down_sample_path+"/"+filename, img_half)


def rotate_imgs(ori_path, rotate_deg):
    dir_path = ori_path+"_rot"+rotate_deg
    check_path(dir_path)

    # Rotate images
    for filename in os.listdir(ori_path):
        image = cv2.imread(ori_path+"/"+filename)
        image_rotate = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(dir_path+"/"+filename, image_rotate)


def images_ipad2colmap(ori_path, out_path):

    gn_file = open(out_path,'w')
    gn_file.writelines("# Image list with two lines of data per image:\n")
    gn_file.writelines("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    gn_file.writelines("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

    IMAGE_ID = 1
    with open(ori_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            elems = line.split(", ")
            
            # timestamp , image_name , x , y , z , qx , qy , qz , qw
            if str(elems[1])=="000000" or str(elems[1])=="frame":
                continue
            tvec = np.array(elems[2:5],dtype=float)
            qw = np.array(elems[8],dtype=float)
            qvec = np.array(elems[5:8],dtype=float)
            image_name = str(elems[1])+".png"
            c2w_matrix = np.array(create_transform_matrix(tvec,qvec,qw))
            inv_c2w_matrix = np.linalg.inv(c2w_matrix)

            # w2c_qvec = [qx, qy, qz, qw]
            w2c_qvec = rotation_matrix_to_quaternion(inv_c2w_matrix[0:3,0:3])
            w2c_tvec = inv_c2w_matrix[0:3,3]

            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            in_str = [str(IMAGE_ID)," ",str(w2c_qvec.item(3))," ",\
                      str(w2c_qvec.item(0))," ",str(w2c_qvec.item(1))," ",str(w2c_qvec.item(2))," ",\
                      str(w2c_tvec[0])," ",str(w2c_tvec[1])," ",str(w2c_tvec[2])," ","1 ",image_name,"\n","1 1 -1\n"]
            
            gn_file.writelines(in_str)
            IMAGE_ID += 1


#   3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
def ply2txt(ori_path, out_path):

    # Read XYZ point cloud from filename PLY file
    plydata = PlyData.read(ori_path)
    pc_list = plydata['vertex'].data.tolist()

    # Insert POINT3D_ID, ERROR, TRACK[]
    counter = 1
    gn_file = open(out_path,'a')
    for i in range(len(pc_list)):
        temp_list = list(pc_list[i][:])
        temp_list.insert(0,counter)
        temp_list.append("1")
        temp_list.append("1")
        temp_list.append("1")
        for j in temp_list:
            gn_file.writelines(str(j)+" ")
        gn_file.writelines("\n")
        counter += 1


def gaussian_noise_bin(ori_path, out_path, rand_a,rand_b):
    print("Still in process")
    colmap_read_model.change_images_binary(ori_path, out_path,rand_a,rand_b)


#ply2txt("T-502-to-503-48964f3a64/T-pointcloud_2.ply", "points3D.txt")
images_ipad2colmap("T-502-to-503-48964f3a64/odometry.csv", "images_v2.txt")
