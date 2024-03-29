'''
File Created: Monday, 15th Jan 2021 1:35:30 pm
Author: Shichen Hu (shichen.hu@tum.de)
I hereby claim that I refered to the OpenCV online documentation for writing this program.
'''
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt 
from utils import save_depth_to_pgm, create_output, read_pfm_file, load_scene_stereo_pair
from stereomideval.structures import MatchData
from stereomideval.dataset import Dataset
from stereomideval.eval import Eval, Timer, Metric




# define paths
# change the DATASET_FOLDER to the right path
matching_method = "SGBM"
scene = "chess1"
DATASET_FOLDER = os.path.join(os.getcwd(),"middlebury2021")
ply_name = "reconstructed.ply"
ply_path = os.path.join(DATASET_FOLDER,
                            scene,
                            ply_name)
left_img, right_img = load_scene_stereo_pair(scene,DATASET_FOLDER, False)
left_img_grayscale = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
left_img_RGB = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
right_img_grayscale = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
right_img_RGB = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

# track the processing time
timer = Timer()
timer.start()

# read in the intrinsic matrix, the follow is the intrinsics for "chess1" scene
K0 = np.array([[1758.23, 0, 953.34],[0, 1758.23, 552.29],[0, 0, 1]])
K1 = np.array([[1758.23, 0, 953.34],[0, 1758.23, 552.29],[0, 0, 1]])


# feature point extraction
orb = cv2.ORB_create()
sift = cv2.SIFT_create()
kp0, dptr0 = sift.detectAndCompute(left_img_grayscale, None)
kp1, dptr1 = sift.detectAndCompute(right_img_grayscale, None)

# comment the following lines to visualize the keypoints
# img = cv2.drawKeypoints(gray1, kp1, None)
# cv2.imshow("Image", img)
# cv2.waitKey(0)

# use brute-force to match keypoints
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# sort the matches using the distance
matches = bf.match(dptr0, dptr1)
matches = sorted(matches, key = lambda x: x.distance)

img3 = cv2.drawMatches(left_img_grayscale, kp0,right_img_grayscale,kp1,matches[:50], right_img_grayscale,flags=2)
# comment the following lines to visualize the matchings
# cv2.imshow("Image", img3)
# cv2.waitKey(0)
pts0 = []
pts1 = []

# extrac the corecponding points from the matchings
for m in matches[:50]:
    idx0 = m.queryIdx
    idx1 = m.trainIdx
    pt0 = kp0[idx0]
    pt1 = kp1[idx1]
    pts0.append(pt0.pt)
    pts1.append(pt1.pt)

pts0 = np.asarray(pts0)
pts1 = np.asarray(pts1)

# using 8 points algorithm to find the FundamentalMat
F, mask = cv2.findFundamentalMat(pts0,pts1, cv2.FM_8POINT)
# compute the essential matrix E
E = K0.T @ F @ K1

# recover the rotation and translation from essential matrix
inlier_num,R,T,mask = cv2.recoverPose(E, pts0, pts1)


distCoeffs0 = np.array([0,0,0,0])
distCoeffs1 = np.array([0,0,0,0])

# rectify the two images
assert(left_img.shape == right_img.shape)
size = [left_img.shape[0], left_img.shape[1]]
R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(K0,distCoeffs0,K1,distCoeffs1,left_img.shape[:2],R,T)
map00,map01 = cv2.initUndistortRectifyMap(K0,distCoeffs0, R0, P0[:,:3],size,cv2.CV_16SC2)
img0_rec = cv2.remap(left_img,map00,map01,cv2.INTER_LINEAR)

map10,map11 = cv2.initUndistortRectifyMap(K1,distCoeffs1, R1, P1[:,:3],size,cv2.CV_16SC2)
img1_rec = cv2.remap(right_img,map10,map11,cv2.INTER_LINEAR)
# cv2.imshow("img1_rec", img1_rec)
# cv2.waitKey(0)

# use StereoSGBM to generate the disparity map
win_size = 11
min_disp = 75
num_disp = 192 # Needs to be divisible by 16

stereo_matching = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    # maxDisparity=max_disp,
    numDisparities = num_disp,
    blockSize = win_size,
    uniquenessRatio = 5,
    speckleWindowSize =5,
    speckleRange = 5,
    disp12MaxDiff = 2,
    P1 = 8*3*win_size**2,#8*3*win_size**2,
    P2 =32*3*win_size**2) #32*3*win_size**2)

disparity = stereo_matching.compute(left_img_grayscale, right_img_grayscale)
disp_img = np.divide(np.float32(disparity), 16.0)
elapsed_time = timer.elapsed()

h, w = 1080.0, 1920.0
# set according to chess1 scene
base = 111.0
focal_length = 1758.23

# Q matrix from the Oreily'book
Q = np.float32([[-1,0,0,w/2.0],
				[0,-1,0,h/2.0],
				[0,0,0,focal_length],
				[0,0,1/base,0]])

points_3D = cv2.reprojectImageTo3D(disp_img, Q)

#Get color points
left_img = left_img.astype('uint8')
colors = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)


#Generate point cloud 
print ("\n Creating the ply file... \n")
create_output(points_3D, colors, ply_path)