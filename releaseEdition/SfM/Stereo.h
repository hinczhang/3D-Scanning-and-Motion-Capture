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
            
            void RTresponse(individualLoader::pairs &left, individualLoader::pairs &right, cv::Mat &R1, cv::Mat &T1, cv::Mat &R2, cv::Mat &T2,std::vector<std::vector<cv::KeyPoint>>& key_points_for_all, std::vector<std::vector<cv::DMatch>>& matches_for_all, std::vector<cv::Point3d>& structure,std::vector<std::vector<int>>& correspond_struct_idx,bool if_first=true);
            void simplified();
        private:
            virtual void patternPointsObtain(cv::Mat &img, cv::Mat &descriptor, std::vector<cv::KeyPoint> &keypoints){}; // virtual function for point search.
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