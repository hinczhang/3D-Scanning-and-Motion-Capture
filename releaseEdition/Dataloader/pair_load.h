#ifndef PAIR_LOAD
#define PAIR_LOAD

// Load the dataset from DTU_dataset

#include<iostream>
#include<vector>
#include<string>
#include<dirent.h>
#include<assert.h>
#include<fstream>
#include<map>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
// define the individual namespace to avoid the conflict
namespace individualLoader{
    
    // the intrisic, extrinsic matrix and fator, depth_min of each image
    struct parameters{
        double intrinsic[3][3];
        double extrinsic[4][4];
        float depth_min;
        float factor;
    };
     
    // one array of image. The struct contains parameters of the reference image. Along with the compared images.
    struct pairs{
        short int ref_img;
        std::vector<short int> src_imgs;
        parameters matrices;
        std::string img_path;
    };
    
    // we perform the data loading by a two-level maps. the first level sets std::string as the index to find a certain scence, the second level
    // sets short int as the index to find a certain image pairs struct.
    class Loader{
        public:
        Loader(std::string path):data_path(path){
            this->loadImagepairs();
        }

        std::map<std::string, std::map<short int, pairs>> pairsInformation();

        private:
            std::string data_path; // the data path
            std::map<std::string, std::map<short int, pairs>> groups; // one data path has several scans, and each scan corresponds to a group
            std::vector<std::string> path_contained(std::string path_name); // find the scan folders in the data path
            void loadImagepairs(); // load the image pairs to the variable groups 
            void split(const std::string& s, std::vector<std::string>& tokens, const std::string& delimiters=" "); // ro simulate the split function like python
            parameters getPrameters(std::string paraPath); // read the parameters of images
            std::string returnEight(short int code); // formulate the number to eight digits
            std::string returnThree(short int code); // formulate the number to eight digits
    };


};


#endif