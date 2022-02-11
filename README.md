# 3D-Scanning-and-Motion-Capture
Mainly for Structure from Motion (SfM) and Multiview Stereo (MVS).

## Debug Edition
Debug edition contains some original test code,containing the full workflow of making the wheel for SfM calculation.  
However, after comparison, we finally choose OpenCV lib as our tool to finish the release edition.  
Even in this kind of situation, I still consider this part would have the educational meaning.  
 1. Fundamential matrix calculation  
 2. Essential matrix calculation  
 3. RANSAC  

## Release Edition
Remember to config the CMakeLists right, as we have not used the regular way to config it.

## Dataset
DTU dataset: https://roboimagedata.compute.dtu.dk/  
Middlebury: https://vision.middlebury.edu/stereo/data/