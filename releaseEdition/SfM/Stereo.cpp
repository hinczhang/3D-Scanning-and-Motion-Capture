#include "Stereo.h"


void stereo::SIFTStereos::patternPointsObtain(cv::Mat &img, cv::Mat &descriptor, std::vector<cv::KeyPoint> &keypoints){
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(400);
    detector->detect(img, keypoints);
    detector->compute(img, keypoints, descriptor);
    cv::Mat draw_img;
    cv::drawKeypoints(img, keypoints, draw_img, cv::Scalar::all(-1));
    cv::imwrite("./test_SIFT.jpg", draw_img);

}

void stereo::ORBStereos::patternPointsObtain(cv::Mat &img, cv::Mat &descriptor, std::vector<cv::KeyPoint> &keypoints){
    cv::Ptr<cv::ORB> detector = cv::ORB::create(400);
    detector->detect(img, keypoints);
    detector->compute(img, keypoints, descriptor);
    descriptor.convertTo(descriptor, 5);
    cv::Mat draw_img;
    cv::drawKeypoints(img, keypoints, draw_img, cv::Scalar::all(-1));
    cv::imwrite("./test_ORB.jpg", draw_img);

}

void stereo::SURFStereos::patternPointsObtain(cv::Mat &img, cv::Mat &descriptor, std::vector<cv::KeyPoint> &keypoints){
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);
    detector->detect(img, keypoints);
    detector->compute(img, keypoints, descriptor);
    cv::Mat draw_img;

    cv::drawKeypoints(img, keypoints, draw_img, cv::Scalar::all(-1));
    cv::imwrite("./test_SURF.jpg", draw_img);

}

// test the output coordinates
void testWrite(std::vector<cv::Point3d> &points, std::string filename="output.xyz"){
    std::ofstream output(filename);
    for(auto point:points){
        output<<std::to_string(point.x)<<" "<<std::to_string(point.y)<<" "<<std::to_string(point.z)<<"\n";

    }
    output.close();
}


void stereo::StereoPairs::RTresponse(individualLoader::pairs &left, individualLoader::pairs &right, cv::Mat &R1, cv::Mat &T1, cv::Mat &R2, cv::Mat &T2,std::vector<std::vector<cv::KeyPoint>>& key_points_for_all, std::vector<std::vector<cv::DMatch>>& matches_for_all, std::vector<cv::Point3d>& structure,std::vector<std::vector<int>>& correspond_struct_idx,bool if_first){
    std::vector<cv::Point2f> leftPoints;
    std::vector<cv::Point2f> rightPoints;
    cv::Mat leftImg = cv::imread(left.img_path, 0);
    cv::Mat rightImg = cv::imread(right.img_path, 0);
    cv::Mat descriptorLeft;
    cv::Mat descriptorRight;
    std::vector<cv::KeyPoint> keypointsLeft;
    std::vector<cv::KeyPoint> keypointsRight;
    this->patternPointsObtain(leftImg, descriptorLeft, keypointsLeft);
    this->patternPointsObtain(rightImg, descriptorRight, keypointsRight);
    
   //cv::FlannBasedMatcher matcher;
    cv::BFMatcher matcher(cv::NORM_L2);

   //std::vector<cv::DMatch> matches;*
   std::vector<std::vector<cv::DMatch>> matches;
   matcher.knnMatch(descriptorLeft, descriptorRight, matches, 2);

   std::vector<cv::DMatch> good_matches;
   // Ratio Test Process
	float min_dist = FLT_MAX;
	for (int r = 0; r < matches.size(); ++r)
	{
		// Rotio Test
		if (matches[r][0].distance > 0.6 * matches[r][1].distance)
		{
			continue;
		}

		float dist = matches[r][0].distance;
		if (dist < min_dist)
		{
			min_dist = dist;
		}
	}

	for (size_t r = 0; r < matches.size(); ++r)
	{

		if (
			matches[r][0].distance > 0.6 * matches[r][1].distance ||
			matches[r][0].distance > 5 * std::max(min_dist, 10.0f)
			)
		{
			continue;
		}

		good_matches.push_back(matches[r][0]);
	}
    // Coordinate handling
    matches_for_all.push_back(good_matches);

    std::vector<int> queryIndex;

    std::vector<int> trainndex;

    for(int i=0;i<good_matches.size();i++){
        queryIndex.push_back(good_matches[i].queryIdx);
        trainndex.push_back(good_matches[i].trainIdx);
    }

    cv::KeyPoint::convert(keypointsLeft,leftPoints,queryIndex);
    cv::KeyPoint::convert(keypointsRight,rightPoints,trainndex);

    cv::Mat K(3,3,CV_64F,left.matrices.intrinsic);
    
    
    if(if_first){
        key_points_for_all.push_back(keypointsLeft);
        key_points_for_all.push_back(keypointsRight);
        std::vector<int> left_idx;
        std::vector<int> right_idx;
        left_idx.resize(keypointsLeft.size(),-1);
        right_idx.resize(keypointsRight.size(),-1);
        cv::Mat mask1,mask2;
        cv::Mat E = cv::findEssentialMat(leftPoints,rightPoints,K,cv::RANSAC,0.999,1.0,mask1);
        
        cv::recoverPose(E,leftPoints,rightPoints,K,R2,T2,mask2);
        std::cout<<"inliers: "<<float(cv::countNonZero(mask1))/leftPoints.size()*100<<"%"<<std::endl;
        this->ProjReconstruct(K,R1,T1,R2,T2,leftPoints,rightPoints,structure);
        correspond_struct_idx.clear();
        
        int idx=0;
        for(int i=0;i<good_matches.size();i++){

            if(mask2.at<uchar>(i)==0) continue;

            left_idx[good_matches[i].queryIdx]=idx;
            right_idx[good_matches[i].trainIdx]=idx;
            idx++;
        }
        correspond_struct_idx.push_back(left_idx);
        correspond_struct_idx.push_back(right_idx);
    }

    else{
        key_points_for_all.push_back(keypointsRight);
        std::vector<int> next_idx;
        next_idx.resize(keypointsRight.size(),-1);
        std::vector<cv::Point3d> object_points;
        std::vector<cv::Point2f> image_points;
        cv::Mat r;

        this->get_objpoints_and_imgpoints(good_matches,correspond_struct_idx[correspond_struct_idx.size()-1],structure,keypointsRight,object_points,image_points);
        if(object_points.size()<=4){
            cv::Mat mask;
            cv::Mat E=cv::findEssentialMat(leftPoints,rightPoints,K,cv::FM_RANSAC,0.9999,1.0,mask);
            std::cout<<"Ex4: inliers: "<<float(cv::countNonZero(mask))/leftPoints.size()*100<<"%"<<std::endl;
            cv::recoverPose(E,leftPoints,rightPoints,K,R2,T2,mask);
        }else{

            cv::solvePnPRansac(object_points,image_points,K,cv::noArray(),r,T2);
            cv::Rodrigues(r,R2);
        }
        
        std::vector<cv::Point3d> next_structure;
        this->ProjReconstruct(K,R1,T1,R2,T2,leftPoints,rightPoints,next_structure);

        this->fusion_structure(good_matches,correspond_struct_idx[correspond_struct_idx.size()-1],next_idx,structure,next_structure);
        correspond_struct_idx.push_back(next_idx);
        
    }

}

void stereo::StereoPairs::ProjReconstruct(cv::Mat& K, cv::Mat& R1, cv::Mat& T1, cv::Mat& R2, cv::Mat& T2, std::vector<cv::Point2f>& p1, std::vector<cv::Point2f>& p2, std::vector<cv::Point3d>& structure)
{
    cv::Mat proj1(3, 4, CV_32FC1);
    cv::Mat proj2(3, 4, CV_32FC1);

    R1.convertTo(proj1(cv::Range(0, 3), cv::Range(0, 3)), CV_32FC1);
    T1.convertTo(proj1.col(3), CV_32FC1);

    R2.convertTo(proj2(cv::Range(0, 3), cv::Range(0, 3)), CV_32FC1);
    T2.convertTo(proj2.col(3), CV_32FC1);

    cv::Mat fK;
    K.convertTo(fK, CV_32FC1);
    proj1 = fK*proj1;
    proj2 = fK*proj2;

    // Triangulation
    cv::Mat s;
    cv::triangulatePoints(proj1, proj2, p1, p2, s);


    structure.reserve(s.cols);
    for (int i = 0; i < s.cols; ++i)
    {
        cv::Mat_<float> col = s.col(i);
        col /= col(3);  // Homogeneous coordinate
        structure.push_back(cv::Point3d(col(0), col(1), col(2)));
    }
}

void stereo::StereoPairs::get_objpoints_and_imgpoints(std::vector<cv::DMatch>& matches,std::vector<int>& struct_indices, std::vector<cv::Point3d>& structure, std::vector<cv::KeyPoint>& key_points,std::vector<cv::Point3d>& object_points,std::vector<cv::Point2f>& image_points){
    object_points.clear();
    image_points.clear();

    for (int i = 0; i < matches.size(); ++i)
    {
        int query_idx = matches[i].queryIdx;
        int train_idx = matches[i].trainIdx;
        int struct_idx = struct_indices[query_idx];
        if (struct_idx < 0) continue;

        object_points.push_back(structure[struct_idx]);
        image_points.push_back(key_points[train_idx].pt);
    }
}


void stereo::StereoPairs::fusion_structure(std::vector<cv::DMatch>& matches, std::vector<int>& struct_indices, std::vector<int>& next_struct_indices,std::vector<cv::Point3d>& structure, std::vector<cv::Point3d>& next_structure)
{
    for (int i = 0; i < matches.size(); ++i)
    {
        int query_idx = matches[i].queryIdx;
        int train_idx = matches[i].trainIdx;

        int struct_idx = struct_indices[query_idx];
        if (struct_idx >= 0) // Keep the same index if the coordinate is the same
        {
            next_struct_indices[train_idx] = struct_idx;
            continue;
        }

        // Add new coordinates
        structure.push_back(next_structure[i]);
        struct_indices[query_idx] = structure.size() - 1;
        next_struct_indices[train_idx] = structure.size() - 1;
    }
}

void stereo::StereoPairs::simplified(){
    std::vector<cv::Point2f> leftPoints;
    std::vector<cv::Point2f> rightPoints;

    cv::Mat leftImg = cv::imread("im0-1.png", 0);
    cv::Mat rightImg = cv::imread("im1-1.png", 0);
    cv::Mat descriptorLeft;
    cv::Mat descriptorRight;
    std::vector<cv::KeyPoint> keypointsLeft;
    std::vector<cv::KeyPoint> keypointsRight;
    this->patternPointsObtain(leftImg, descriptorLeft, keypointsLeft);
    this->patternPointsObtain(rightImg, descriptorRight, keypointsRight);
    //cv::FlannBasedMatcher matcher;
    cv::BFMatcher matcher(cv::NORM_L2);
    //std::vector<cv::DMatch> matches;*
    std::vector<std::vector<cv::DMatch>> matches;
    //matcher.match(descriptorLeft, descriptorRight, matches);
    matcher.knnMatch(descriptorLeft, descriptorRight, matches, 2);

    std::vector<cv::DMatch> good_matches;

	float min_dist = FLT_MAX;
	for (int r = 0; r < matches.size(); ++r)
	{
		// Rotio Test
		if (matches[r][0].distance > 0.6 * matches[r][1].distance)
		{
			continue;
		}

		float dist = matches[r][0].distance;
		if (dist < min_dist)
		{
			min_dist = dist;
		}
	}

	for (size_t r = 0; r < matches.size(); ++r)
	{
		if (
			matches[r][0].distance > 0.6 * matches[r][1].distance ||
			matches[r][0].distance > 5 * std::max(min_dist, 10.0f)
			)
		{
			continue;
		}

		good_matches.push_back(matches[r][0]);
	}


    std::vector<int> queryIndex;

    std::vector<int> trainndex;

    for(int i=0;i<good_matches.size();i++){
        queryIndex.push_back(good_matches[i].queryIdx);
        trainndex.push_back(good_matches[i].trainIdx);
    }

    cv::KeyPoint::convert(keypointsLeft,leftPoints,queryIndex);
    cv::KeyPoint::convert(keypointsRight,rightPoints,trainndex);

    cv::Mat R1,R2,T1,T2;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1733.74,0,792.27,0,1733.74,541.89,0,0,1);


    R1 = (cv::Mat_<double>(3, 3) << 1,0,0,0,1,0,0,0,1);
    T1 = (cv::Mat_<double>(3, 1) << 0,0,0);    
    cv::Mat distCoefL= (cv::Mat_<double>(5, 1) << 0.0, 0.0,0.0, 0.0, 0.0);
    cv::Mat distCoefR= (cv::Mat_<double>(5, 1) << 0.0, 0.0,0.0, 0.0, 0.0);

    cv::Mat mask1,mask2;
    cv::Mat tempdistL = distCoefL.t();
    cv::Mat tempdistR = distCoefR.t();
    cv::Mat E = cv::findEssentialMat(leftPoints,rightPoints,K,cv::RANSAC,0.999,1.0,mask1);
    
    cv::recoverPose(E,leftPoints,rightPoints,K,R2,T2,mask2);

    std::cout<<"inliers: "<<float(cv::countNonZero(mask1))/leftPoints.size()*100<<"%"<<std::endl;
    std::vector<cv::Point3d> structure;
    
    //this->ProjReconstruct(K,R1,T1,R2,T2,leftPoints,rightPoints,structure);

    int sgbmWinSize =  5;
    int NumDisparities = (int((257-33)/16)+1)*16;
    int UniquenessRatio = 6;
    
    cv::Mat Rl, Rr, Pl, Pr, Q; 
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(33,NumDisparities,sgbmWinSize);
    


    cv::Mat R_vector;
    cv::Rodrigues(R2,R_vector);

    cv::Rect leftRec,rightRec;

    

    cv::stereoRectify(K,distCoefL,K,distCoefR,leftImg.size(),R_vector,T2,Rl,Rr,Pl,Pr,Q,cv::CALIB_ZERO_DISPARITY,0,leftImg.size(),&leftRec,&rightRec);  //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
    cv::Mat mapLx, mapLy, mapRx, mapRy;     //映射表
    cv::initUndistortRectifyMap(K, distCoefL, Rl, Pl, leftImg.size(), CV_32FC1, mapLx, mapLy);
    cv::initUndistortRectifyMap(K, distCoefR, Rr, Pr, rightImg.size(), CV_32FC1, mapRx, mapRy);

    cv::Mat rectifyImageL, rectifyImageR;
    cv::remap(leftImg, rectifyImageL, mapLx, mapLy, cv::INTER_LINEAR);//INTER_LINEAR
    cv::imwrite("recConyL.png", rectifyImageL);
    cv::remap(rightImg, rectifyImageR, mapRx, mapRy, cv::INTER_LINEAR);//INTER_LINEAR
    cv::imwrite("recConyR.png", rectifyImageR);
    cv::Mat disp, dispf, disp8;
    
  /*
    cv::stereoRectify(K1,distCoefL,K2,distCoefR,leftImg.size(),R_vector,T2,Rl,Rr,Pl,Pr,Q,cv::CALIB_ZERO_DISPARITY,0,leftImg.size(),&leftRec,&rightRec);  //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
    cv::Mat mapLx, mapLy, mapRx, mapRy;     //映射表
    cv::initUndistortRectifyMap(K1, distCoefL, Rl, Pl, leftImg.size(), CV_32FC1, mapLx, mapLy);
    cv::initUndistortRectifyMap(K2, distCoefR, Rr, Pr, rightImg.size(), CV_32FC1, mapRx, mapRy);

    cv::Mat rectifyImageL, rectifyImageR;
    cv::remap(leftImg, rectifyImageL, mapLx, mapLy, cv::INTER_LINEAR);//INTER_LINEAR
    cv::imwrite("recConyL.png", rectifyImageL);
    cv::remap(rightImg, rectifyImageR, mapRx, mapRy, cv::INTER_LINEAR);//INTER_LINEAR
    cv::imwrite("recConyR.png", rectifyImageR);
    cv::Mat disp, dispf, disp8;
  */

    int cn = rectifyImageL.channels();

    sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);

    sgbm->setUniquenessRatio(UniquenessRatio);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(9);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
    sgbm->compute(rectifyImageL,rectifyImageR,disp);

    
    cv::Mat img1p, img2p;
    cv::copyMakeBorder(rectifyImageL, img1p, 0, 0, NumDisparities, 0,1);

    cv::copyMakeBorder(rectifyImageR, img2p, 0, 0, NumDisparities, 0, 1);
    dispf = disp.colRange(NumDisparities, img2p.cols- NumDisparities);
    
    cv::Mat xyz;
    dispf.convertTo(disp8, CV_8U, 255 / (NumDisparities *16.));
    cv::imwrite("disparity.png", disp8);
   
    reprojectImageTo3D(disp, xyz, Q, true); 
    xyz = xyz * 16;
    
    cv::Mat colors = cv::imread("im0-1.png", 1);
    const double max_z = 16.0e4;
    std::ofstream outModel("new_3d.ply");

    outModel<<"ply\nformat ascii 1.0\nelement vertex 1748468\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\nend_header\n";

    for(int y=0;y<xyz.rows;y++){
        for(int x=0;x<xyz.cols;x++){
            cv::Vec3f point=xyz.at<cv::Vec3f>(y,x);
            cv::Vec3b color_info = colors.at<cv::Vec3b>(y,x);
            if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            outModel<<std::to_string(point[0])<<" "<<std::to_string(point[1])<<" "<<std::to_string(point[2])<<" "<<std::to_string(color_info[2])<<" "<<std::to_string(color_info[1])<<" "<<std::to_string(color_info[0])<<" 0"<<"\n";
        }
    }

    outModel.close();

}

