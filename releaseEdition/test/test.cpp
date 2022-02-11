
#include "Stereo.h"
#include"mv_stereo.h"
int main(){
    // test the data loader
    
    individualLoader::Loader myloader("/3d/datasets/dtu_testing/dtu/");
    std::map<std::string, std::map<short int, individualLoader::pairs>> steroInitial = myloader.pairsInformation();

    
    stereo::StereoPairs *pair = new stereo::SURFStereos(steroInitial["scan1"]);


    std_mvs::MVSMatching *matching = new std_mvs::MVSMatching(steroInitial["scan1"]);
    matching->run();
    cv::Ptr<cv::xfeatures2d::HarrisLaplaceFeatureDetector> detector=cv::xfeatures2d::HarrisLaplaceFeatureDetector::create();

    return 0;
}