project(Dataloader)

cmake_minimum_required(VERSION 3.5)
aux_source_directory(. DIR_LOADER_SRCS)
add_library(Dataloader ${DIR_LOADER_SRCS})

set("OpenCV_DIR" "/3d/libs/archive/opencv4-source/build")
find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED xfeatures2d) 
include_directories( ${OpenCV_INCLUDE_DIRS} )
