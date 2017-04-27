This is a test implementation program for an augmented reality social game that we envisioned; this implementation focuses
on face mask and face detection primarily and how openCV can be used to achieve these results. 

research into the library allowed us to successfully detect faces and render a 2D sprite onto this detect face.
However complications arise in making a 3d face mask that stays on the face and isn't just placed on top of the face.
For this we hypothize using 3D telemetry of our camera to define world space and using this to put together the area on the 
detected face with defined coordinates to a 3D mask. 

Calculating camera calibration and testing this on aruco markers was successful. However using this on a detected face proved
more difficult than expected. Face morphing the other technique discussed in the presentation proved to be ineffective with live 
video capture. 

We hope to better this implementation by further improving our understanding of the methods and techniques in use with OpenCV.
OpenCV is a very powerful Computer Vision library. 

Note about the main function-
to calibrate camera uncommit calibrateCameraCall function, commit rest:
*press space bar to take picture 
*press enter when atleast 15 pictures have been taken to start camera calibration- information will be saved to textfile "CameraCalibrated"
*press escape to exit

umcommit arucoDetection and loadCameraCalibration to check worldspace aruco marker implementation, commit rest:
*press excape to exit. Place multiple aruco markers to see results after camera calibration

uncommit faceMasking to see results of face masking, commit rest:
*press "x" to shift between facedetect view and facemask view
*press enter in facemask view to filter through different mask sprites
*press escape to exit

By Uttkarsh Narayan, August Orlow, Shubham Gupta and Alexandre Dias

GitHub:https://github.com/uttkarsh21/BGE_FinalProject_FaceFilter/tree/master/BGE_FaceFilter/BGE_FaceFilter
