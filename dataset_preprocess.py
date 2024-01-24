from preprocess_utils import check_path, motion_blur, vid2imgs, downsample, rotate_imgs, gaussian_noise_bin
from preprocess_utils import images_ipad2colmap, ply2txt


direction       = "horizontal"      # Motion blur direction ["horizontal", "vertical"]
kernel_size     = 10                # Motion blur kernel size
rotate_deg      = 90                # Image Rotation degree

folder_path     = "T-502-to-503-48964f3a64"
ply_path        = folder_path + "/T-pointcloud_2.ply"
odometry_path   = folder_path + "/odometry.csv"
vid_path        = folder_path + "/rgb.mp4"
img_path        = folder_path + "/images"

if __name__ == '__main__':
    print("Select mode (1-6):")
    print("1. Motion blur")
    print("2. Video to images")
    print("3. Downsampling images")
    print("4. Rotate images")
    print("5. Convert from .ply to point3d.txt format")
    print("6. Convert from iPad odometry.txt to llff images.txt format")
    print("-------------")
    mode = input("Mode: ")
    ori_path = "images"
    print("\nSelcted mode ",mode,", procecssing...\n")

    # Add motion blur to images
    if mode == "1":
        motion_blur(ori_path, direction, kernel_size)

    # Chop video to images
    elif mode == "2":
        check_path(img_path)
        vid2imgs(vid_path, img_path)

    # Downsampling images
    elif mode == "3":
        down_sample_factor = input("Downsampling factor: ")
        if down_sample_factor.is_integer():
            if int(down_sample_factor) % 2 == 0:
                downsample(ori_path, down_sample_factor)
            else:
                print("ERROR: Downsampling factor should be the multiples of 2")
        else:
            print("ERROR: Downsampling factor should be integer")

    # Rotate images
    elif mode == "4":
        rotate_imgs(ori_path, rotate_deg)

    # Convert from .ply to point3d.txt format
    elif mode == "5":
        ori_path = ply_path
        check_path(folder_path + "/0")
        out_path = folder_path + "/0/points3D.txt"
        ply2txt(ori_path, out_path)

    # Convert from ipad odometry .csv to llff .txt format
    elif mode == "6":
        ori_path = odometry_path
        check_path(folder_path + "/0")
        out_path = folder_path + "/0/images.txt"
        images_ipad2colmap(ori_path, out_path)

    # Undesignated situation
    else:
        print("Mode undesignated.\n")
    print("Finished")
