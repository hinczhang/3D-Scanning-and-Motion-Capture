project(SfM)

aux_source_directory(. DIRSRCS)


include_directories(../Dataloader)

cmake_minimum_required(VERSION 3.5)
aux_source_directory(. DIR_SFM_SRCS)
add_library(SfM ${DIR_SFM_SRCS})

set("OpenCV_DIR" "/3d/libs/archive/opencv4-source/build")
find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED xfeatures2d) 


include_directories( ${OpenCV_INCLUDE_DIRS} )
set(CMAKE_CXX_STANDARD 11)
