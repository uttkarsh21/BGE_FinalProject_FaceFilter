#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

double min_face_size = 20;
double max_face_size = 200;
Mat frame;

int main()
{
	VideoCapture cap(0);
	namedWindow("window1", 1);

	if (!cap.isOpened())  // check if we succeeded
		return -1;

	while (1)
	{
		Mat frame;
		cap >> frame;

		frame = detectFace(frame);

		imshow("window1", frame);
		// Press 'c' to escape
		if (waitKey(1) == 'c') break;
	}

	waitKey(0);
	return 0;
}

Mat detectFace(Mat image)
{

	// Load Face cascade (.xml file)
	CascadeClassifier face_cascade;
	face_cascade.load("C:/Users/Uttkarsh/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");

	// Detect faces
	std::vector<Rect> faces;

	face_cascade.detectMultiScale(image, faces, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size), Size(max_face_size, max_face_size));

	// Draw circles on the detected faces
	for (int i = 0; i < faces.size(); i++)
	{   // Lets only track the first face, i.e. face[0] 
		min_face_size = faces[0].width*0.7;
		max_face_size = faces[0].width*1.5;
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);


		ellipse(image, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

	}
	return image;
}
