#ifndef STEREO_PAIR
#define STEREO_PAIR
#include "pair_load.h"
#include<opencv2/features2d.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>
#include<opencv2/core/affine.hpp>
#include<opencv2/calib3d.hpp>

/*
TODO: Implement matrix first, and then use RANSAC
*/

namespace stereo{
// several image pairs are reuqired (individualLoader::pairs)
// pre handling
// pattern examination
// matching
// geommetry correction

    class StereoPairs{
        /*
        To do: the geometric correction and extrinsic parameters calculation
        */
        public:
            StereoPairs(std::map<short int, individualLoader::pairs> pairs):images_array(pairs){

            }; // load the data loader
            
            void plotMatcher_test(int left_num=0,int right_num=0); // test the match effect. it should not be applied in the release edition.
            void RTresponse(individualLoader::pairs &left, individualLoader::pairs &right, cv::Mat &R1, cv::Mat &T1, cv::Mat &R2, cv::Mat &T2,std::vector<std::vector<cv::KeyPoint>>& key_points_for_all, std::vector<std::vector<cv::DMatch>>& matches_for_all, std::vector<cv::Point3d>& structure,std::vector<std::vector<int>>& correspond_struct_idx,bool if_first=true);
            void simplified();
        private:
            void mather(cv::Mat descriptorLeft, cv::Mat descriptorRight); // the release edition of matcher.
            virtual void patternPointsObtain(cv::Mat &img, cv::Mat &descriptor, std::vector<cv::KeyPoint> &keypoints){}; // virtual function for point search.
            cv::Mat matrixFundation(std::vector<cv::Point2f> &leftPoints, std::vector<cv::Point2f> &rightPoints);
            cv::Mat matrixEssential(cv::Mat &F, individualLoader::parameters &left, individualLoader::parameters &right);
            //void poseEstimation(cv::Mat &E);
            cv::Mat normalize(cv::Mat &scalar);
            void reconstruct(cv::Mat &entrinsic, cv::Mat& K, cv::Mat& R, cv::Mat& T, std::vector<cv::Point2f>& p1, std::vector<cv::Point2f>& p2, std::vector<cv::Point3d>& release3dPoint, int left_num=0, int right_num=0); //for 3D points
            cv::Mat RANSAC(std::vector<cv::Point2f> &leftPoints, std::vector<cv::Point2f> &rightPoints, int maxIterNum, int minrequiredNum, double threshold, int assertNum);
            double calc_sampson_distance(cv::Mat &F, cv::Point2f &left, cv::Point2f &right);
            
        // Increment development
            void ProjReconstruct(cv::Mat& K, cv::Mat& R1, cv::Mat& T1, cv::Mat& R2, cv::Mat& T2, std::vector<cv::Point2f>& p1, std::vector<cv::Point2f>& p2, std::vector<cv::Point3d>& structure);       
            void get_objpoints_and_imgpoints(std::vector<cv::DMatch>& matches,std::vector<int>& struct_indices, std::vector<cv::Point3d>& structure, std::vector<cv::KeyPoint>& key_points,std::vector<cv::Point3d>& object_points,std::vector<cv::Point2f>& image_points);
            void fusion_structure(std::vector<cv::DMatch>& matches, std::vector<int>& struct_indices, std::vector<int>& next_struct_indices,std::vector<cv::Point3d>& structure, std::vector<cv::Point3d>& next_structure);
        protected:
            std::map<short int, individualLoader::pairs> images_array; // data loader
            
            
    };

    // SIFT
    class SIFTStereos: public StereoPairs{
        public:
            SIFTStereos(std::map<short int, individualLoader::pairs> pairs):StereoPairs(pairs){

            }
        private:
            void patternPointsObtain(cv::Mat &img, cv::Mat &descriptor, std::vector<cv::KeyPoint> &keypoints);
            
    };

    // ORB
    class ORBStereos: public StereoPairs{
        public:
            ORBStereos(std::map<short int, individualLoader::pairs> pairs):StereoPairs(pairs){

            }
        private:
            void patternPointsObtain(cv::Mat &img, cv::Mat &descriptor, std::vector<cv::KeyPoint> &keypoints);
    };


    //SURF
    class SURFStereos: public StereoPairs{
        public:
            SURFStereos(std::map<short int, individualLoader::pairs> pairs):StereoPairs(pairs){

            }
        private:
            void patternPointsObtain(cv::Mat &img, cv::Mat &descriptor, std::vector<cv::KeyPoint> &keypoints);
    };

};

#endif