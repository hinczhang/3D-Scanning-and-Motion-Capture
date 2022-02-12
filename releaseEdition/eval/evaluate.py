'''
File Created: Monday, 15th Jan 2021 1:35:30 pm
Author: Shichen Hu (shichen.hu@tum.de)
'''
import os
from tkinter import CENTER
import cv2
import numpy as np
from tqdm import tqdm
from utils import load_scene_stereo_pair, load_cal, load_scene_disparity, create_output
from stereomideval.eval import Timer, Eval, Metric
from stereomideval.structures import MatchData
from stereomideval.dataset import Dataset
# import open3d
from matplotlib import pyplot as plt 

# point this to the middlebury 2021 dataset folder
DATASET_FOLDER = os.path.join(os.getcwd(),"middlebury2021")
scene_list = os.listdir(DATASET_FOLDER)

# flags for evalludating keypoints or dense matching
EVALUATE_KEYPOINT = True
EVALUATE_DENSE_MATCHING = False

def match(kp_left, dptr_left, kp_right, dptr_right, method="flann", detect="orb"):
    """Keypoints matching by flann or bf"""
    if method == "flann":
        if detect != "orb":
            # KD tree indexing
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(dptr_left, dptr_right, 2)
        else:
            # locality sensitive hashing
            index_params = dict(algorithm=6,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=2
                            )
            search_params = {}
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(dptr_left, dptr_right, k=2)

        
    elif method == "bf":
        matches = cv2.BFMatcher().knnMatch(dptr_left, dptr_right, k=2)

    # lowe's ratio test:
    ratio_thresh = 0.7
    good_matches = []
    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return good_matches

def extract_points(left_img, right_img, method):
    "keypoints extraction by ORB, SIFT or SURF"
    if method == "orb":
        method_fn = cv2.ORB_create()
    elif method == "sift":
        method_fn = cv2.SIFT_create()
    elif method == "surf":
        method_fn = cv2.xfeatures2d.SURF_create()
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    kp_left, dptr_left = method_fn.detectAndCompute(gray_left, None)
    kp_right, dptr1_right = method_fn.detectAndCompute(gray_right, None)
    return kp_left, dptr_left, kp_right, dptr1_right
    

def MatchingsToF(matches, kp0, kp1, K):
    """Recover fundamental matrix from matching"""
    pts0, pts1 = [], []
    for m in matches:
        idx0 = m.queryIdx
        idx1 = m.trainIdx
        pt0 = kp0[idx0]
        pt1 = kp1[idx1]
        pts0.append(pt0.pt)
        pts1.append(pt1.pt)

    pts0 = np.asarray(pts0)
    pts1 = np.asarray(pts1)

    E, mask = cv2.findEssentialMat(pts0,pts1, K, cv2.FM_RANSAC)
    num_inliers = (mask == 1).sum()
    num_outliers = (mask == 0).sum()
    return num_inliers, num_outliers

def evaluate_keypoints(detector=["orb","sift"], matcher="bf"):
    "print out the statistics for keypoint evaluation"
    if "orb" in detector:
        orb_list= []
    if "sift" in detector:
        sift_list = []
    if "surf" in detector:
        surf_list = []
    total_inliers = 0
    total_outliers = 0
    print("Evaluating point extractions:")
    timer = Timer()
    timer.start()
    for scene in tqdm(scene_list):
        # load the left and right images
        left_img, right_img = load_scene_stereo_pair(scene,DATASET_FOLDER, False)
        camera_info = load_cal(scene,DATASET_FOLDER)
        c_x, c_y, f = camera_info.c_x, camera_info.c_y, camera_info.focal_length
        K = np.float32([[f,0,c_x],
                        [0,f,c_y],
                        [0,0,1]])
        if "orb" in detector:
            kp_left, dptr_left, kp_right, dptr_right = extract_points(left_img, right_img, "orb")
            orb_list.append(len(kp_left))
            orb_list.append(len(kp_right))
            matches = match(kp_left, dptr_left, kp_right, dptr_right, matcher, "orb")
            i, o = MatchingsToF(matches, kp_left, kp_right, K)
            total_inliers += i 
            total_outliers += o
        if "sift" in detector:
            kp_left, dptr_left, kp_right, dptr_right = extract_points(left_img, right_img, "sift")
            sift_list.append(len(kp_left))
            sift_list.append(len(kp_right))
            matches = match(kp_left, dptr_left, kp_right, dptr_right, matcher, "sift")
            i, o = MatchingsToF(matches, kp_left, kp_right, K)
            total_inliers += i 
            total_outliers += o
        if "surf" in detector:
            kp_left, dptr_left, kp_right, dptr_right = extract_points(left_img, right_img, "surf")
            surf_list.append(len(kp_left))
            surf_list.append(len(kp_right))
            matches = match(kp_left, dptr_left, kp_right, dptr_right, matcher, "surf")
            i, o = MatchingsToF(matches, kp_left, kp_right, K)
            total_inliers += i 
            total_outliers += o
            
        

    elapsed_time = timer.elapsed()
    print(f"average processing time per scene: {elapsed_time/len(scene_list) :.6f}")
    # print results for matching method:
    print(f"[{matcher}]: outlier portion is {total_outliers / (total_inliers + total_outliers):.6f}")

    if "orb" in detector:
        print(f"[ORB] average points found {np.array(orb_list).mean()}")
    if "sift" in detector:
        print(f"[SIFT] average points found {np.array(sift_list).mean()}")
    if "surf" in detector:
        print(f"[SURF] average points found {np.array(surf_list).mean()}")





def evaluate_disparity(mvs_method="SGBM", display_images=False, generate_gt_pc=False, generate_pc=False, gt=False, save_disp=False, save_depth=False):
    """print out the evaluation metrics and store the generated depth/disparity maps and pointcloud"""
    timer = Timer()
    timer.start()
    bad_scores = []
    for scene in tqdm(scene_list):
        if scene != "traproom1":
            continue
        # load the left and right images
        left_img, right_img = load_scene_stereo_pair(scene,DATASET_FOLDER, False)
        left_img_grayscale = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        left_img_RGB = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img_grayscale = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        right_img_RGB = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        ply_name = f'reconstructed_{mvs_method}.ply' f'reconstructed_{mvs_method}_gt.ply'
        gt_ply_name = f'reconstructed_{mvs_method}_gt.ply'
        disp_name = f"disp_image.png"
        depth_name = f"depth.png"
        gt_depth_name = f"depth_gt.png"
        gt_disp_name = f"disp_image_gt.png"
        depth_path = os.path.join(DATASET_FOLDER,
                                scene,
                                depth_name)
        gt_depth_path = os.path.join(DATASET_FOLDER,
                                scene,
                                gt_depth_name)
        disp_path = os.path.join(DATASET_FOLDER,
                                scene,
                                disp_name)
        gt_disp_path = os.path.join(DATASET_FOLDER,
                                scene,
                                gt_disp_name)
        ply_path = os.path.join(DATASET_FOLDER,
                                scene,
                                ply_name)
        gt_ply_path = os.path.join(DATASET_FOLDER,
                                scene,
                                gt_ply_name)


        camera_info = load_cal(scene,DATASET_FOLDER)
        c_x, c_y, f = camera_info.c_x, camera_info.c_y, camera_info.focal_length
        K = np.float32([[f,0,c_x],
                        [0,f,c_y],
                        [0,0,1]])

        # load the gt disparity map and calculate gt depth
        gt_disp = load_scene_disparity(scene, DATASET_FOLDER, display_images, save_path=gt_disp_path)
        gt_depth = Dataset.disp_to_depth(gt_disp, camera_info.focal_length,
                                                camera_info.doffs, camera_info.baseline)
        cv2.imshow("depth", gt_depth)
        cv2.waitKey(0)
        # plt.imshow(gt_depth)
        # plt.show()
        
        if generate_gt_pc:
            rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(open3d.geometry.Image(left_img_RGB), open3d.geometry.Image(gt_depth), convert_rgb_to_intensity=False)
            intrinsic = open3d.camera.PinholeCameraIntrinsic(int(camera_info.width), int(camera_info.height), f, f, c_x, c_y)
            pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            open3d.io.write_point_cloud(gt_ply_path, pcd)

        win_size = 13
        min_disp = 0
        num_disp = int((camera_info.ndisp // 16) * 16)
        if mvs_method == "StereoBM":
            stereo_matching = cv2.StereoBM_create(numDisparities=num_disp, blockSize=13)
        #Create Block matching object.
        elif mvs_method == "SGBM": 
            stereo_matching = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities = num_disp,
                blockSize = win_size,
                uniquenessRatio = 5,
                speckleWindowSize =5,
                speckleRange = 5,
                disp12MaxDiff = 2,
                P1 = 8*3*win_size**2,#8*3*win_size**2,
                P2 =32*3*win_size**2 #32*3*win_size**2
            )
        disparity = stereo_matching.compute(left_img_grayscale, right_img_grayscale)
        depth = Dataset.disp_to_depth(np.divide(np.float32(disparity), 16.0), camera_info.focal_length,
                                                camera_info.doffs, camera_info.baseline)
        # cv2.imshow("depth", depth)
        # cv2.waitKey(0)
        # plt.imshow(depth)
        # plt.show()
        if display_images:
            disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, \
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_resize = cv2.resize(disp_norm, dsize=(0, 0), fx=0.2, fy=0.2)
            cv2.imshow('image', cv2.applyColorMap(disp_resize, cv2.COLORMAP_JET))
            cv2.waitKey(0)
            
        if save_disp:
            disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, \
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_resize = cv2.resize(disp_norm, dsize=(0, 0), fx=0.2, fy=0.2)
            cv2.imwrite(disp_path, cv2.applyColorMap(disp_resize, cv2.COLORMAP_JET))
        
        if save_depth:
            cv2.imwrite(depth_path,depth.astype(np.uint16))
            cv2.imwrite(gt_depth_path,gt_depth.astype(np.uint16))
            

        w, h = camera_info.width, camera_info.height
        base = camera_info.baseline
        focal_length = camera_info.focal_length
        Q = np.float32([[-1,0,0,w/2.0],
                        [0,-1,0,h/2.0],
                        [0,0,0,focal_length],
                        [0,0,1/base,0]])

        gt_disp_image = np.divide(np.float32(gt_disp), 16.0)
        disp_image = np.divide(np.float32(disparity), 16.0)
        points_3D = cv2.reprojectImageTo3D(disp_image, Q) if gt==False else cv2.reprojectImageTo3D(gt_disp_image, Q)

        left_img = left_img.astype('uint8')
        colors = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

        mask_map = disp_image > (disp_image.min()+2)

        #Mask colors and points. 
        output_points = points_3D[mask_map]
        output_colors = colors[mask_map]
        if generate_pc:
            create_output(points_3D, colors, ply_path)
        match_result = MatchData.MatchResult(
        left_img,right_img,gt_disp,disp_image,0,camera_info.ndisp)
        # Evalulate match results against all Middlebury metrics
        metric_result = Eval.eval_metric(Metric.bad200,match_result)
        bad_scores.append(metric_result.result)

    elapsed_time = timer.elapsed()
    print(f"average processing time per scene: {elapsed_time/len(scene_list) :.6f}")
    # print results for matching method:
    print(f"average bad200 score: {np.array(bad_scores).mean()}")
    
    

def main():
    # evaluate keypoint detection and matching
    if EVALUATE_KEYPOINT:
        evaluate_keypoints(detector=["sift"], matcher="flann")
        evaluate_keypoints(detector=["sift"], matcher="bf")
        evaluate_keypoints(detector=["orb"], matcher="flann")
        evaluate_keypoints(detector=["orb"], matcher="bf")
        evaluate_keypoints(detector=["surf"], matcher="flann")
        evaluate_keypoints(detector=["surf"], matcher="bf")
    if EVALUATE_DENSE_MATCHING:
        evaluate_disparity(mvs_method="SGBM", generate_pc=False, gt=False, save_disp=False, save_depth=False)

    # evaluate disparity maps

if __name__ == "__main__":
    main()

    
