
#include "Stereo.h"
#include"mv_stereo.h"
int main(){
    // test the data loader
    
    individualLoader::Loader myloader("/3d/datasets/dtu_testing/dtu/");
    std::map<std::string, std::map<short int, individualLoader::pairs>> steroInitial = myloader.pairsInformation();
    
/*
    for(iter=steroInitial.begin();iter!=steroInitial.end();iter++){
        
            std::cout<<iter->first<<" "<<std::endl;

            std::map<short int, individualLoader::pairs>::iterator i;
            for(i=iter->second.begin();i!=iter->second.end();i++){
                std::cout<<i->first<<std::endl;
            }
    };
*/

    
stereo::StereoPairs *pair = new stereo::SURFStereos(steroInitial["scan1"]);
//cv::Mat img = imread("/3d/datasets/dtu_testing/dtu/scan1/images/00000000.jpg", cv::IMREAD_GRAYSCALE);

for(int i=1;i<steroInitial["scan1"][0].src_imgs.size();i++){
//    pair->plotMatcher_test(i-1,i);
}
std_mvs::MVSMatching *matching = new std_mvs::MVSMatching(steroInitial["scan1"]);
matching->run();
cv::Ptr<cv::xfeatures2d::HarrisLaplaceFeatureDetector> detector=cv::xfeatures2d::HarrisLaplaceFeatureDetector::create();
/*
std::vector<cv::KeyPoint> keypoints;

detector->detect(img,keypoints);
*/
    return 0;
}