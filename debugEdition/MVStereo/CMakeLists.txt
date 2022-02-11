project(MVStereo)

aux_source_directory(. DIRSRCS)


include_directories(../Dataloader)

include_directories(../Stereo)

cmake_minimum_required(VERSION 3.5)
aux_source_directory(. DIR_SFM_SRCS)
add_library(MVStereo ${DIR_SFM_SRCS})

set("OpenCV_DIR" "/3d/libs/archive/opencv4-source/build")
#set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
#set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to installed Ceres")
#set(glog_DIR "/3d/libs/glog/build")


#find_package(Eigen3 REQUIRED)
#find_package(glog REQUIRED)
#find_package(Ceres REQUIRED)



find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED xfeatures2d) 


include_directories( ${OpenCV_INCLUDE_DIRS} )
set(CMAKE_CXX_STANDARD 11)