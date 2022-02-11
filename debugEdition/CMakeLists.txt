project(SfM)


aux_source_directory(. DIRSRCS)


add_subdirectory(Dataloader)
include_directories(./Dataloader)

add_subdirectory(Stereo)
include_directories(./Stereo)

add_subdirectory(MVStereo)
include_directories(./MVStereo)

cmake_minimum_required(VERSION 3.5)

set("OpenCV_DIR" "/3d/libs/archive/opencv4-source/build")
find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED xfeatures2d)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-std=c++14")

add_executable( test ./test/test.cpp)
target_link_libraries( test ${OpenCV_LIBS} Dataloader Stereo MVStereo ${CERES_LIBRARIES})

