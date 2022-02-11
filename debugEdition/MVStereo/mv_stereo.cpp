#include"mv_stereo.h"


void Write(std::vector<cv::Point3d> &points, std::string filename="output.xyz"){
    std::ofstream output(filename);
    for(auto point:points){
        output<<std::to_string(point.x)<<" "<<std::to_string(point.y)<<" "<<std::to_string(point.z)<<"\n";

    }
    output.close();
}

void std_mvs::MVSMatching::test_mvs(){
	

}

void std_mvs::MVSMatching::run(){
    stereo::StereoPairs *pair = new stereo::SURFStereos(this->images_array);
	//pair->simplified();
	    
    std::vector<std::vector<cv::KeyPoint>> key_points_for_all;
    std::vector<std::vector<cv::DMatch>> matches_for_all;
    std::vector<std::vector<int>> idx_for_all;
    std::vector<cv::Mat> Rs;
    std::vector<cv::Mat> Ts;

    std::vector<cv::Point3d> structure;

    // initial
    Rs.push_back(cv::Mat::eye(3,3,CV_32FC1));
    Ts.push_back(cv::Mat::zeros(3,1,CV_32FC1));
    Rs.push_back(cv::Mat::eye(3,3,CV_32FC1));
    Ts.push_back(cv::Mat::zeros(3,1,CV_32FC1));

    pair->RTresponse(images_array[0],images_array[images_array[0].src_imgs[0]],Rs[0],Ts[0],Rs[1],Ts[1],key_points_for_all,matches_for_all,structure,idx_for_all,true);

	std::vector<individualLoader::pairs> dealImagesArray;
	dealImagesArray.push_back(images_array[0]);
	dealImagesArray.push_back(images_array[images_array[0].src_imgs[0]]);
	std::map<int,bool> dict;
	dict.insert(std::pair<int, bool>(0,true));
	dict.insert(std::pair<int, bool>(images_array[0].src_imgs[0],true));
	int image = 19;
	while(true){
		if(dealImagesArray.size()>image)break;
		int temp_idx = 0;
		int length = dealImagesArray[dealImagesArray.size()-1].src_imgs.size();
		bool flag = true;
		for(int x=0;x<length;x++){
			temp_idx = dealImagesArray[dealImagesArray.size()-1].src_imgs[x];
			if(dict.find(temp_idx)!=dict.end()) continue;
			else {flag=false;break;};
		}

		if(flag)break;
		
		dict.insert(std::pair<int, bool>(temp_idx,true));
		dealImagesArray.push_back(images_array[temp_idx]);
	}
	
	
	

    for(int i=1;i<dealImagesArray.size()-1;i++){

		cv::Mat tempR, tempT;
        pair->RTresponse(dealImagesArray[i],dealImagesArray[i+1],Rs[Rs.size()-1],Ts[Ts.size()-1],tempR,tempT,key_points_for_all,matches_for_all,structure,idx_for_all,false);
		Rs.push_back(tempR);
        Ts.push_back(tempT);

    }

    Write(structure);

    cv::Mat K(3,3,CV_64F,images_array[0].matrices.intrinsic);
    cv::Mat intrinsic(cv::Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
    std::vector<cv::Mat> extrinsics;
    for (size_t i = 0; i < Rs.size(); ++i){
        cv::Mat extrinsic(6, 1, CV_64FC1);
        cv::Mat r;
        cv::Rodrigues(Rs[i], r);

        r.copyTo(extrinsic.rowRange(0, 3));
        Ts[i].copyTo(extrinsic.rowRange(3, 6));

        extrinsics.push_back(extrinsic);
    }


    this->bundle_adjustment(intrinsic, extrinsics, idx_for_all, key_points_for_all, structure);

}

void std_mvs::MVSMatching::bundle_adjustment(cv::Mat& intrinsic,std::vector<cv::Mat>& extrinsics, std::vector<std::vector<int>>& correspond_struct_idx,std::vector<std::vector<cv::KeyPoint>>& key_points_for_all,std::vector<cv::Point3d>& structure)
{
	ceres::Problem problem;

	// load extrinsics (rotations and motions)
	for (size_t i = 0; i < extrinsics.size(); ++i)
	{
		problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);
	}
	// fix the first camera.
	problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());

	// load intrinsic
	problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // load fx, fy, cx, cy

	// load points
	ceres::LossFunction* loss_function = new ceres::HuberLoss(4);	// loss function make bundle adjustment robuster.
	for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
	{
		std::vector<int> &point3d_ids = correspond_struct_idx[img_idx];

		std::vector<cv::KeyPoint> &key_points = key_points_for_all[img_idx];
		for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
		{
			int point3d_id = point3d_ids[point_idx];
			if (point3d_id < 0)
				continue;

			cv::Point2d observed = key_points[point_idx].pt;
			// The first parameter is for the type of the cost function, the second is the dimensional of residual, the left are coefficients' dimensions.
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<std_mvs::ReprojectCost, 2, 4, 6, 3>(new std_mvs::ReprojectCost(observed));

			problem.AddResidualBlock(
				cost_function,
				loss_function,
				intrinsic.ptr<double>(),			// Intrinsic
				extrinsics[img_idx].ptr<double>(),	// View Rotation and Translation
				&(structure[point3d_id].x)			// Point in 3D space
			);
		}
	}

	// Solve BA
	ceres::Solver::Options ceres_config_options;
	ceres_config_options.minimizer_progress_to_stdout = false;
	ceres_config_options.logging_type = ceres::SILENT;
	ceres_config_options.num_threads = 1;
	ceres_config_options.preconditioner_type = ceres::JACOBI;
	ceres_config_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

	ceres::Solver::Summary summary;
	ceres::Solve(ceres_config_options, &problem, &summary);

	if (!summary.IsSolutionUsable())
	{
		std::cout << "Bundle Adjustment failed." << std::endl;
	}
	else
	{
		// Display statistics about the minimization
		std::cout << std::endl
			<< "Bundle Adjustment statistics (approximated RMSE):\n"
			<< " #views: " << extrinsics.size() << "\n"
			<< " #residuals: " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
			<< " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
			<< " Time (s): " << summary.total_time_in_seconds << "\n"
			<< std::endl;
		std::cout << summary.BriefReport() << "\n";
	}
}
