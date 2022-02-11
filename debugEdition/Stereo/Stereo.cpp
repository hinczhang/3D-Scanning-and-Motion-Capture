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



void stereo::StereoPairs::plotMatcher_test(int left_num, int right_num){
    individualLoader::pairs left = this->images_array[left_num];
    individualLoader::pairs right = this->images_array[this->images_array[left_num].src_imgs[right_num]];
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

   //std::vector<cv::DMatch> matches;
   std::vector<std::vector<cv::DMatch>> matches;
   //matcher.match(descriptorLeft, descriptorRight, matches);
   matcher.knnMatch(descriptorLeft, descriptorRight, matches, 2);

   std::vector<cv::DMatch> good_matches;
   // 获取满足Ratio Test的最小匹配的距离
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
		// 排除不满足Ratio Test的点和匹配距离过大的点
		if (
			matches[r][0].distance > 0.6 * matches[r][1].distance ||
			matches[r][0].distance > 5 * std::max(min_dist, 10.0f)
			)
		{
			continue;
		}

		// 保存匹配点
		good_matches.push_back(matches[r][0]);
	}

    //-- Draw only "good" matches
    cv::Mat img_matches;
    cv::drawMatches( leftImg, keypointsLeft, rightImg, keypointsRight,
               good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::imwrite("match.jpg", img_matches);

    // Coordinate handling

    std::vector<cv::Point2f> leftPoints;
    std::vector<int> queryIndex;
    std::vector<cv::Point2f> rightPoints;
    std::vector<int> trainndex;

    for(int i=0;i<good_matches.size();i++){
        queryIndex.push_back(good_matches[i].queryIdx);
        trainndex.push_back(good_matches[i].trainIdx);
    }

    cv::KeyPoint::convert(keypointsLeft,leftPoints,queryIndex);
    cv::KeyPoint::convert(keypointsRight,rightPoints,trainndex);


    cv::Mat F8 = cv::findFundamentalMat(leftPoints,rightPoints,cv::FM_8POINT);
    cv::Mat F8_mine = this->matrixFundation(leftPoints,rightPoints);
    cv::Mat F=cv::findFundamentalMat(leftPoints,rightPoints,cv::FM_RANSAC);
    cv::Mat E = this->matrixEssential(F, left.matrices, right.matrices);

    cv::Mat K(3,3,CV_64F,left.matrices.intrinsic);
    cv::Mat R, t;
    cv::Mat mask;
    
    E=cv::findEssentialMat(leftPoints,rightPoints,K,cv::FM_RANSAC,0.9999,1.0,mask);

    std::cout<<"inlier: "<<float(cv::countNonZero(mask))/leftPoints.size()<<"\n";
    int pass=cv::recoverPose(E, leftPoints, rightPoints, K, R, t);
    std::cout<<"front: "<<float(pass)/leftPoints.size()<<"\n";
    std::vector<cv::Point3d> releasePoints;


   cv::Mat D1,D2;

   cv::Mat leftE(4,4,CV_64F,left.matrices.extrinsic);
   cv::Mat rightE(4,4,CV_64F,right.matrices.extrinsic);
   cv::Mat trans(3,4,CV_32F);
   R.convertTo(trans(cv::Range(0, 3), cv::Range(0, 3)), CV_32FC1);
   t.convertTo(trans.col(3), CV_32FC1);

    this->reconstruct(leftE,K,R,t,leftPoints,rightPoints,releasePoints, left_num, right_num);

    cv::Mat leftColor = cv::imread(left.img_path,0);
    cv::Mat rightColor = cv::imread(right.img_path,0);
   

    //this->denseReconstruction(K,R,t,leftColor,rightColor,std::min(left.matrices.depth_min,right.matrices.depth_min),std::max(left.matrices.factor,right.matrices.factor));
}

void stereo::StereoPairs::mather(cv::Mat descriptorLeft, cv::Mat descriptorRight){

}

cv::Mat stereo::StereoPairs::normalize(cv::Mat &scalar){
    double meanX=0;
    double meanY=0;
    for(int i=0;i<scalar.rows;i++){
        meanX+=scalar.at<double>(i,0);
        meanY+=scalar.at<double>(i,1);
    }
    meanX/=scalar.rows;
    meanY/=scalar.rows;

    double sigma=0;
    for(int i=0;i<scalar.rows;i++){
        sigma+=sqrt((pow(scalar.at<double>(i,0)-meanX,2)+pow(scalar.at<double>(i,1)-meanY,2)));
    }
    sigma = sigma/scalar.rows/sqrt(2);
    cv::Mat T = cv::Mat::zeros(3,3,CV_64F);
    T.at<double>(0,0)=1/sigma;
    T.at<double>(0,2)=-meanX/sigma;
    T.at<double>(1,1)=1/sigma;
    T.at<double>(1,2)=-meanY/sigma;
    T.at<double>(2,2)=1;
    return T;

}

cv::Mat stereo::StereoPairs::matrixFundation(std::vector<cv::Point2f> &leftPoints, std::vector<cv::Point2f> &rightPoints){
    cv::Mat A;
    // initialize the parameter matrix
    cv::Mat x1 = cv::Mat::ones(leftPoints.size(),3,CV_64F);
    cv::Mat x2 = cv::Mat::ones(rightPoints.size(),3,CV_64F);

    for(int i=0;i<leftPoints.size();i++){
        x1.at<double>(i,0) = leftPoints[i].x;
        x1.at<double>(i,1) = leftPoints[i].y;
        x2.at<double>(i,0) = rightPoints[i].x;
        x2.at<double>(i,1) = rightPoints[i].y;
    }

    cv::Mat T1 = this->normalize(x1);

    cv::Mat T2 = this->normalize(x2);




    x1=x1*cv::Mat(T1.t());
    x2=x2*cv::Mat(T2.t()); 
    A = x1.colRange(0,1).mul(x2.colRange(0,1));
    
    cv::hconcat(A,x1.colRange(1,2).mul(x2.colRange(0,1)),A);
    cv::hconcat(A,x2.colRange(0,1),A);
    cv::hconcat(A,x1.colRange(0,1).mul(x2.colRange(1,2)),A);
    cv::hconcat(A,x1.colRange(1,2).mul(x2.colRange(1,2)),A);
    cv::hconcat(A,x2.colRange(1,2),A);
    cv::hconcat(A,x1.colRange(0,1),A);
    cv::hconcat(A,x1.colRange(1,2),A);
    cv::hconcat(A,cv::Mat::ones(leftPoints.size(),1,CV_64F),A);
    cv::Mat w, u, vt;
    cv::SVDecomp(A,w,u,vt,cv::SVD::FULL_UV);
    cv::Mat f=vt.rowRange(8,9).clone();

    cv::Mat F = f.reshape(0,3);

    cv::Mat Fw, Fu, Fvt;
    cv::SVD::compute(F, Fw, Fu, Fvt,cv::SVD::FULL_UV);
    
    Fw.at<double>(2,0) = 0.0f;
    cv::Mat FwDiagnal = cv::Mat::zeros(3,3,CV_64F);
    FwDiagnal.at<double>(0,0) = Fw.at<double>(0,0); FwDiagnal.at<double>(1,1) = Fw.at<double>(1,0);
    F = Fu*FwDiagnal*Fvt;


    F=cv::Mat(T2.t())*F*T1;

    return F/F.at<double>(2,2);

}

cv::Mat stereo::StereoPairs::matrixEssential(cv::Mat &F, individualLoader::parameters &left, individualLoader::parameters &right){
    cv::Mat leftIntri(3,3,CV_64F,left.intrinsic);
    cv::Mat rightIntri(3,3, CV_64F, right.intrinsic);

    cv::Mat Ehat = rightIntri.t()*F*leftIntri.inv();
    cv::Mat EhatU, EhatW, EhatVt;
    cv::SVD::compute(Ehat, EhatW, EhatU, EhatVt);
    
    
    
    cv::Mat EhatWnew = cv::Mat::zeros(3,3,CV_64F);
    // first method
    
    //EhatWnew.at<double>(0,0) = 1;EhatWnew.at<double>(1,1) = 1;

    // second method
    //EhatWnew.at<double>(0,0) = (EhatW.at<double>(0,0)+EhatW.at<double>(0,1))/2;EhatWnew.at<double>(1,1) = EhatWnew.at<double>(0,0);
    EhatWnew.at<double>(0,0) = 1;EhatWnew.at<double>(1,1) = 1;
    cv::Mat E = EhatU*EhatWnew*EhatVt;
    return E;

    
}


void testWrite(std::vector<cv::Point3d> &points, std::string filename="output.xyz"){
    std::ofstream output(filename);
    for(auto point:points){
        output<<std::to_string(point.x)<<" "<<std::to_string(point.y)<<" "<<std::to_string(point.z)<<"\n";

    }
    output.close();
}

void stereo::StereoPairs::reconstruct(cv::Mat &entrinsic, cv::Mat& K, cv::Mat& R, cv::Mat& T, std::vector<cv::Point2f>& p1, std::vector<cv::Point2f>& p2, std::vector<cv::Point3d>& release3dPoint, int left_num, int right_num){
	//triangulatePoints only support float type?
	cv::Mat proj1(3, 4, CV_32FC1);
	cv::Mat proj2(3, 4, CV_32FC1);

	proj1(cv::Range(0, 3), cv::Range(0, 3)) = cv::Mat::eye(3, 3, CV_32FC1);
	proj1.col(3) = cv::Mat::zeros(3, 1, CV_32FC1);
	R.convertTo(proj2(cv::Range(0, 3), cv::Range(0, 3)), CV_32FC1);
	T.convertTo(proj2.col(3), CV_32FC1);

	cv::Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK*proj1;
	proj2 = fK*proj2;


    cv::Mat s;
	cv::triangulatePoints(proj1, proj2, p1, p2, s);
    release3dPoint.reserve(s.cols);
    for (int i = 0; i < s.cols; ++i)
	{
		cv::Mat_<float> col = s.col(i);
		col /= col(3);	
		release3dPoint.push_back(cv::Point3d(col(0), col(1), col(2)));
	}
    testWrite(release3dPoint,std::to_string(left_num)+"_"+std::to_string(right_num)+"N.xyz");

}

double stereo::StereoPairs::calc_sampson_distance(cv::Mat &F, cv::Point2f &left, cv::Point2f &right){
    double p2_F_p1 = 0.0;
    p2_F_p1 += right.x * (left.x * F.at<double>(0,0) + left.y * F.at<double>(0,1) + F.at<double>(0,2));
    p2_F_p1 += right.y * (left.x * F.at<double>(1,0) + left.y * F.at<double>(1,1) + F.at<double>(1,2));
    p2_F_p1 +=     1.0 * (left.x * F.at<double>(2,0) + left.y * F.at<double>(2,1) + F.at<double>(2,2));
    p2_F_p1 *= p2_F_p1;
    double sum = 1.0e-6;
    sum += pow(left.x * F.at<double>(0,0) + left.y * F.at<double>(0,1) + F.at<double>(0,2), 2);
    sum += pow(left.x * F.at<double>(1,0) + left.y * F.at<double>(1,1) + F.at<double>(1,2), 2);
    sum += pow(right.x * F.at<double>(0,0) + right.y * F.at<double>(1,0) + F.at<double>(2,0), 2);
    sum += pow(right.x * F.at<double>(0,1) + right.y * F.at<double>(1,1) + F.at<double>(2,1), 2);

    return p2_F_p1 / sum;
}

cv::Mat stereo::StereoPairs::RANSAC(std::vector<cv::Point2f> &leftPoints, std::vector<cv::Point2f> &rightPoints, int maxIterNum, int minrequiredNum, double threshold, int assertNum){
    cv::Mat bestFit;
    float bestErr = FLT_MAX;
    while(true){
    int iterations = 0;
    
    
    std::vector<int> indexes;
    for(int i=0;i<leftPoints.size();i++){
        indexes.push_back(i);
    }

    while(iterations<maxIterNum){
        std::vector<cv::Point2f> templeft;
        std::vector<cv::Point2f> tempright;
        
        // maybeInliers
        std::srand(unsigned(std::time(NULL)));
        //shuffle the data randomly 
        std::random_shuffle(indexes.begin(),indexes.end());

        for(int i=0;i<indexes.size();i++){
            templeft.push_back(leftPoints[indexes[i]]);
            tempright.push_back(rightPoints[indexes[i]]);
        }

        std::vector<cv::Point2f> assignedLeft;
        std::vector<cv::Point2f> assignedRight;
        
        assignedLeft.assign(templeft.begin(),templeft.begin()+minrequiredNum);
        assignedRight.assign(tempright.begin(),tempright.begin()+minrequiredNum);

        // erase used data
        templeft.erase(templeft.begin(),templeft.begin()+minrequiredNum);
        tempright.erase(tempright.begin(),tempright.begin()+minrequiredNum);

        cv::Mat maybemodel=this->matrixFundation(assignedLeft,assignedRight);
        // alsoInliers
        std::vector<cv::Point2f> alsoInliersLeft;
        std::vector<cv::Point2f> alsoInliersRight;

        for(int i=0; i<templeft.size();i++){
            cv::Mat leftCoord = (cv::Mat_<double>(3,1)<<templeft[i].x,templeft[i].y,1.0);
            cv::Mat rightCoord = (cv::Mat_<double>(3,1)<<tempright[i].x,tempright[i].y,1.0);
            cv::Mat error = leftCoord.t()*maybemodel*rightCoord;
            double disErr = this->calc_sampson_distance(maybemodel,templeft[i],tempright[i]);
            if(disErr<=0.5){
                alsoInliersLeft.push_back(templeft[i]);
                alsoInliersRight.push_back(tempright[i]);
            }
        }

        if(alsoInliersRight.size()>assertNum){
        //if(true){
            std::vector<cv::Point2f> suitCollectionLeft;
            std::vector<cv::Point2f> suitCollectionRight;
            suitCollectionLeft.insert(suitCollectionLeft.end(),assignedLeft.begin(),assignedLeft.end());
            suitCollectionLeft.insert(suitCollectionLeft.end(),alsoInliersLeft.begin(),alsoInliersLeft.end());
            suitCollectionRight.insert(suitCollectionRight.end(),assignedRight.begin(),assignedRight.end());
            suitCollectionRight.insert(suitCollectionRight.end(),alsoInliersRight.begin(),alsoInliersRight.end());

            cv::Mat betterModel = this->matrixFundation(suitCollectionLeft,suitCollectionRight);

            double thisErr=0.0;
            for(int i=0;i<suitCollectionRight.size();i++){
                cv::Mat leftCoord = (cv::Mat_<double>(3,1)<<suitCollectionLeft[i].x,suitCollectionLeft[i].y,1.0);
                cv::Mat rightCoord = (cv::Mat_<double>(3,1)<<suitCollectionRight[i].x,suitCollectionRight[i].y,1.0);
                cv::Mat error = leftCoord.t()*maybemodel*rightCoord;
                //thisErr += abs(error.at<double>(0,0));
                double disErr = this->calc_sampson_distance(maybemodel,templeft[i],tempright[i]);
                thisErr += (disErr<=1?1:0);
            }
            thisErr/=suitCollectionRight.size();
            if(thisErr<bestErr){
                bestFit = betterModel;
                bestErr = thisErr;
            }
        }
        iterations++;
    }

    
    if(bestFit.rows==0){
        if(assertNum<0.1*leftPoints.size()){
            return cv::Mat();
        }
        else{
            assertNum=assertNum*0.8;
            
        }

    }
    else{
        break;
    }
    
    }
    double ratioInliers=0.0;
    for(int i=0; i<leftPoints.size();i++){
        cv::Mat leftCoord = (cv::Mat_<double>(3,1)<<leftPoints[i].x,leftPoints[i].y,1.0);
        cv::Mat rightCoord = (cv::Mat_<double>(3,1)<<rightPoints[i].x,rightPoints[i].y,1.0);
        cv::Mat error = leftCoord.t()*bestFit*rightCoord;
        if(abs(error.at<double>(0,0))<threshold){
            ratioInliers+=1.0;
        }
    }
    std::cout<<"My own algorithm inliers: "<<ratioInliers/leftPoints.size()<<std::endl;
    return bestFit;
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
   //matcher.match(descriptorLeft, descriptorRight, matches);
   matcher.knnMatch(descriptorLeft, descriptorRight, matches, 2);

   std::vector<cv::DMatch> good_matches;
   // 获取满足Ratio Test的最小匹配的距离
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
		// 排除不满足Ratio Test的点和匹配距离过大的点
		if (
			matches[r][0].distance > 0.6 * matches[r][1].distance ||
			matches[r][0].distance > 5 * std::max(min_dist, 10.0f)
			)
		{
			continue;
		}

		// 保存匹配点
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
    //两个相机的投影矩阵[R T]，triangulatePoints只支持float型
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

    //三角重建
    cv::Mat s;
    cv::triangulatePoints(proj1, proj2, p1, p2, s);


    structure.reserve(s.cols);
    for (int i = 0; i < s.cols; ++i)
    {
        cv::Mat_<float> col = s.col(i);
        col /= col(3);  //齐次坐标，需要除以最后一个元素才是真正的坐标值
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
        //if(query_idx>=struct_indices.size())continue;
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
        if (struct_idx >= 0) //若该点在空间中已经存在，则这对匹配点对应的空间点应该是同一个，索引要相同
        {
            next_struct_indices[train_idx] = struct_idx;
            continue;
        }

        //若该点在空间中已经存在，将该点加入到结构中，且这对匹配点的空间点索引都为新加入的点的索引
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
    // 获取满足Ratio Test的最小匹配的距离
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
		// 排除不满足Ratio Test的点和匹配距离过大的点
		if (
			matches[r][0].distance > 0.6 * matches[r][1].distance ||
			matches[r][0].distance > 5 * std::max(min_dist, 10.0f)
			)
		{
			continue;
		}

		// 保存匹配点
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

    int sgbmWinSize =  5;//根据实际情况自己设定
    int NumDisparities = (int((257-33)/16)+1)*16;//根据实际情况自己设定
    int UniquenessRatio = 6;//根据实际情况自己设定
    
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

    //outModel<<"";
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

