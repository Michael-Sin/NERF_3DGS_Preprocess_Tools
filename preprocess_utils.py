import cv2
import os
import numpy as np 
import colmap_read_model
from plyfile import PlyData
import numpy as np


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
    t = np.matrix.trace(transform_matrix)
    m = transform_matrix[0:3, 0:3]
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    # q = [qw, qx, qy, qz]
    if(t > 0):
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5/t
        q[0] = (m[2,1] - m[1,2]) * t
        q[1] = (m[0,2] - m[2,0]) * t
        q[2] = (m[1,0] - m[0,1]) * t
    else:
        #print(transform_matrix)
        i = 0
        if (m[1,1] > m[0,0]):
            i = 1
        if (m[2,2] > m[i,i]):
            i = 2
        j = (i+1)%3
        k = (j+1)%3
        t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k,j] - m[j,k]) * t
        q[j] = (m[j,i] + m[i,j]) * t
        q[k] = (m[k,i] + m[i,k]) * t
    return q



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
            image_name = str(elems[1])+".jpg"
            c2w_matrix = np.array(create_transform_matrix(tvec,qvec,qw))
            inv_c2w_matrix = np.linalg.inv(c2w_matrix)

            # w2c_qvec = [qw, qx, qy, qz]
            w2c_qvec = rotation_matrix_to_quaternion(inv_c2w_matrix)

            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            in_str = [str(IMAGE_ID)," ",str(w2c_qvec.item(0))," ",\
                      str(w2c_qvec.item(1))," ",str(w2c_qvec.item(2))," ",str(w2c_qvec.item(3))," ",\
                      str(tvec[0])," ",str(tvec[1])," ",str(tvec[2])," ","1 ",image_name,"\n","1 1 -1\n"]
            
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
