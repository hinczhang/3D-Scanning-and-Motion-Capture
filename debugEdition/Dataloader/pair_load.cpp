#include "pair_load.h"

std::vector<std::string> individualLoader::Loader::path_contained(std::string path_name){
    std::vector<std::string> names;
    DIR *dir;
    struct dirent *ent;
    dir = opendir (path_name.c_str());
    assert(dir!=nullptr);
    while ((ent = readdir (dir)) != NULL) {
        if(std::string(ent->d_name).substr(0,4)=="scan")
            names.push_back(std::string(ent->d_name));
    }
    closedir (dir);
    return names;
}

void individualLoader::Loader::loadImagepairs(){
    std::vector<std::string> names = this->path_contained(this->data_path);
    for(auto name:names){
        std::map<short int, individualLoader::pairs> singleGroup;
        std::string filename = this->data_path+name+"/pair.txt";
        std::ifstream input_pair(filename);
        int pairsNum = 0;
        std::string line;
        getline(input_pair,line);
        pairsNum = atoi(line.c_str());

        while(pairsNum){
            individualLoader::pairs photoPair;
            getline(input_pair,line);
            photoPair.ref_img = atoi(line.c_str());
            
            photoPair.matrices = this->getPrameters(this->data_path+name+"/cams/"+this->returnEight(photoPair.ref_img)+"_cam.txt");
            photoPair.img_path = this->data_path+name+"/images/"+this->returnEight(photoPair.ref_img)+".jpg";
            //photoPair.img_path = "/3d/datasets/dtu_training/mvs_training/dtu/Rectified/scan1_train/rect_"+this->returnThree(photoPair.ref_img+1)+"_6_r5000.png";
            getline(input_pair,line);
            std::vector<std::string> tokens;
            split(line,tokens," ");
            int flag = 0;
            for(auto x:tokens){
                if(flag%2==1)
                    photoPair.src_imgs.push_back(atoi(x.c_str()));
                flag++;
            }
            singleGroup.insert(std::pair<short int, individualLoader::pairs>(photoPair.ref_img,photoPair));
            pairsNum--;
        }

        input_pair.close();
        this->groups.insert(std::pair<std::string, std::map<short int, individualLoader::pairs>>(name,singleGroup));
        singleGroup.clear();


    }
}

void individualLoader::Loader::split(const std::string& s, std::vector<std::string>& tokens, const std::string& delimiters)
{
    std::string::size_type lastPos=s.find_first_not_of(delimiters,0);
    std::string::size_type pos=s.find_first_of(delimiters,lastPos);
    while(lastPos!=std::string::npos || pos!=std::string::npos)
    {
        tokens.push_back(s.substr(lastPos,pos-lastPos));
        lastPos=s.find_first_not_of(delimiters,pos);
        pos=s.find_first_of(delimiters,lastPos);
    } 
}

std::map<std::string, std::map<short int, individualLoader::pairs>> individualLoader::Loader::pairsInformation(){
    return this->groups;
}

individualLoader::parameters individualLoader::Loader::getPrameters(std::string paraPath){
    individualLoader::parameters para;

    std::ifstream inFile(paraPath);

    std::string number;
    std::string line;
    
    // eliminate the extrinsic line
    inFile>>line;
    for(int i=0;i<16;i++){
        inFile>>number;
        para.extrinsic[int(i/4)][i-int(i/4)*4] = atof(number.c_str());
    }

    inFile>>line;
    for(int i=0;i<9;i++){
        inFile>>number;
        para.intrinsic[int(i/3)][i-int(i/3)*3] = atof(number.c_str());
    }

    inFile>>number;
    para.depth_min = atof(number.c_str());
    inFile>>number;
    para.factor = atof(number.c_str());

    //std::cout<<para.extrinsic<<std::endl;
    //std::cout<<para.intrinsic<<std::endl;

    inFile.close();
    return para;

}

std::string individualLoader::Loader::returnEight(short int code){
    std::string stringCode = std::to_string(code);
    while(stringCode.size()<8){
        stringCode = "0"+stringCode;
    }
    return stringCode;
}


std::string individualLoader::Loader::returnThree(short int code){
    std::string stringCode = std::to_string(code);
    while(stringCode.size()<3){
        stringCode = "0"+stringCode;
    }
    return stringCode;
}