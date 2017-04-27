#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/aruco.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <stdio.h>
#include <fstream>

using namespace std;
using namespace cv;

const float calibrationSquareDimension = 0.025f; //length of calibration squares.
const float arucoSquareDimension = 0.08f; //in meters
const Size chessboardDimensions = Size(6, 9);

int input = 2;
int maskInput = 1;
int fps = 30;

double min_face_size = 20;
double max_face_size = 200;


Mat frame;
Mat mask;

//function to create known position on chessboard based measured length of squares in board and contrast difference between different squares
void createKnownBoardPosition(Size boardSize, float squareEdgeLenght, vector<Point3f>& corners)
{
	for (int i= 0; i<boardSize.height; i++)
	{
		for (int j = 0; j<boardSize.width; j++ )
		{
			corners.push_back(Point3f(j*squareEdgeLenght, i*squareEdgeLenght, 0.0f));
		}
	}
}

//mark in view on the chessboard the corners of the square
void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, Size(9, 6), pointBuf,CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);

		if (found)
		{
			allFoundCorners.push_back(pointBuf);
		}

		if (showResults)
		{
			drawChessboardCorners(*iter, Size(9, 6), pointBuf, found);
			imshow("looking for corners", *iter);
			waitKey(0);
		}
	}
}

//to create aruco markers to be printed out later
void createArucoMarkers()
{
	Mat outputMarker;
	Ptr<aruco::Dictionary>markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);
	for (int i = 0; i < 50; i++)
	{
		aruco::drawMarker(markerDictionary, i, 500, outputMarker, 1);
		ostringstream convert;
		string imageName = "4x4Marker_";
		convert << imageName << i << ".jpg";
		imwrite(convert.str(), outputMarker);

	}
}

//superimposing a mask sprite onto the detected face from teh detectFace function
Mat putMask(Mat src, Point center, Size face_size)
{
	Mat mask1, src1;
	resize(mask, mask1, face_size);

	// ROI selection
	Rect roi(center.x - face_size.width / 2, center.y - face_size.width / 2, face_size.width, face_size.width);
	src(roi).copyTo(src1);

	// to make the white region transparent
	Mat mask2, m, m1;
	cvtColor(mask1, mask2, CV_BGR2GRAY);
	threshold(mask2, mask2, 230, 255, CV_THRESH_BINARY_INV);

	vector<Mat> maskChannels(3), result_mask(3);
	split(mask1, maskChannels);
	bitwise_and(maskChannels[0], mask2, result_mask[0]);
	bitwise_and(maskChannels[1], mask2, result_mask[1]);
	bitwise_and(maskChannels[2], mask2, result_mask[2]);
	merge(result_mask, m);        

	mask2 = 255 - mask2;
	vector<Mat> srcChannels(3);
	split(src1, srcChannels);
	bitwise_and(srcChannels[0], mask2, result_mask[0]);
	bitwise_and(srcChannels[1], mask2, result_mask[1]);
	bitwise_and(srcChannels[2], mask2, result_mask[2]);
	merge(result_mask, m1);       

	addWeighted(m, 1, m1, 1, 0, m1);   

	m1.copyTo(src(roi));

	return src;
}

//detecting a frontal face using Haar cascades which uses the viola jones object detection framework
Mat detectFace(Mat image)
{

	// Load Face cascade (.xml file)
	CascadeClassifier face_cascade, eye_cascade;
	face_cascade.load("C:/Users/Uttkarsh/opencv/install/etc/haarcascades/haarcascade_frontalface_alt2.xml");
	
	// Detect faces
	std::vector<Rect> faces;

	face_cascade.detectMultiScale(image, faces, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size), Size(max_face_size, max_face_size));

	// Draw blocks on the detected faces
	for (int i = 0; i < faces.size(); i++)
	{   // Lets only track the first face, i.e. face[0] 
		min_face_size = faces[0].width*0.7;
		max_face_size = faces[0].width*1.5;
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		Point centerDetect(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		Point start(faces[i].x, faces[i].y);
			switch (input)
			{
			case 1:		image = putMask(image, center, Size(faces[i].width, faces[i].height));
						break;
			case 2:		rectangle(image, start, centerDetect, Scalar(255, 0, 255), 4, 8, 0);
						break;
			default:	break;
			}
	}
	return image;
}

//updates the camera matrix and distance coefficients i.e camera calibration
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float  squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients)
{
	vector<vector<Point2f>> checkerboardImageSpacePoints;
	getChessboardCorners(calibrationImages, checkerboardImageSpacePoints, false);

	vector<vector<Point3f>> worldSpaceCornerPoints(1);

	createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);

	worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

	vector<Mat> rVectors, tVectors;

	distanceCoefficients = Mat::zeros(8, 1, CV_64F);

	calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);

}

//save camera calibration in a text file "CameraCalibrated"
bool savedCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients)
{
	ofstream outStream;
	outStream.open(name);
	if (outStream)
	{
		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;

		outStream << rows << endl;
		outStream << columns << endl;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c <columns; c++)
			{
				double value = cameraMatrix.at<double>(r, c);
				outStream << value << endl;
			}
		}

		rows = distanceCoefficients.rows;
		columns = distanceCoefficients.cols;

		outStream << rows << endl;
		outStream << columns << endl;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c <columns; c++)
			{
				double value = distanceCoefficients.at<double>(r, c);
				outStream << value << endl;
			}
		}

		outStream.close();
		return true;
	}
	return false;
}

//load data from the "cameraCalibrated" text file which holds camera calibration data
bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients)
{
	ifstream inStream;
	inStream.open(name);
	if (inStream)
	{
		uint16_t rows;
		uint16_t columns;
		
		//for camera coefficients
		inStream >> rows;
		inStream >> columns;

		cameraMatrix = Mat(Size(columns, rows), CV_64F);

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double read = 0.0f;
				inStream >> read;
				cameraMatrix.at<double>(r, c) = read;
				cout << cameraMatrix.at<double>(r, c) << "\n";
			}
		}

		//for distance coefficients
		inStream >> rows;
		inStream >> columns;

		distanceCoefficients = Mat::zeros(rows, columns, CV_64F);

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double read = 0.0f;
				inStream >> read;
				distanceCoefficients.at<double>(r, c) = read;
				cout << distanceCoefficients.at<double>(r, c) << "\n";
			}
		}
		inStream.close();
		return true;
	}
	return false;
}

//to handle actually taking pictures of the chessboard so that camera can be calibrated and the data saved
void calibrateCameraCall(string webcam, Mat& cameraMatrix, Mat& distanceCoefficients)
{

	Mat drawToFrame;

	vector<Mat> savedImages;

	vector<vector<Point2f>> markerCorners, rejectedCandidates;

	VideoCapture vid(0);

	if (!vid.isOpened())
	{
		return;
	}

	
	namedWindow(webcam, CV_WINDOW_AUTOSIZE);

	while (1)
	{
		if (!vid.read(frame))
			break;

		vector<Vec2f> foundPoints;
		bool found = false;

		found = findChessboardCorners(frame, chessboardDimensions, foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		frame.copyTo(drawToFrame);

		drawChessboardCorners(drawToFrame, chessboardDimensions, foundPoints, found);

		if (found)
			imshow(webcam, drawToFrame);
		else
			imshow(webcam, frame);

		char character = waitKey(1000 / fps);

		switch (character)
		{
		case ' ':
			//press space bar to saving image
			if (found)
			{
				Mat temp;
				frame.copyTo(temp);
				savedImages.push_back(temp);
			}
			break;
		case 13:
			//press enter to start clibration
			if (savedImages.size() > 15)
			{
				cameraCalibration(savedImages, chessboardDimensions, calibrationSquareDimension, cameraMatrix, distanceCoefficients);
				savedCameraCalibration("cameraCalibrated", cameraMatrix, distanceCoefficients);
			}
			break;
		case 27:
			//press escape key to exit
			return;
			break;

		}

	}
}

//handles the face masking and face detect features
int faceMasking(string webcam, const Mat& cameraMatrix, const Mat& distanceCoefficients)
{
	VideoCapture vid(0);

	mask = imread("C:/Users/Uttkarsh/Documents/masks/11.jpg");

	if (!vid.isOpened())
	{
		return -1;
	}

	namedWindow(webcam, CV_WINDOW_AUTOSIZE);

	while (1)
	{
		if (!vid.read(frame))
			break;

		frame = detectFace(frame);
		imshow(webcam, frame);

		char character = waitKey(1000 / fps);

	
		switch (character)
		{
		case 'x':
			if (input == 1)
				input = 2;
			else
				input = 1;
			break;
		case 13:
			//press enter to shift between masks
			if (maskInput == 1)
			{
				maskInput = 2;
				mask = imread("C:/Users/Uttkarsh/Documents/masks/11.jpg");
			}
			else
				if (maskInput == 2)
				{
					maskInput = 3;
					mask = imread("C:/Users/Uttkarsh/Documents/masks/3.jpg");
				}
				else
					if (maskInput == 3)
					{
						maskInput = 1;
						mask = imread("C:/Users/Uttkarsh/Documents/masks/5.jpg");
					}
			break;
		case 27:
			//press escape key to exit
			return 1;
			break;

		}
		
	}
	return 1;
}

//handles aruco detection
int arucoDetection(string webcam, const Mat& cameraMatrix, const Mat& distanceCoefficients)
{

	vector<int> markerIds;
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	aruco::DetectorParameters parameters;

	Ptr< aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

	VideoCapture vid(0);

	if (!vid.isOpened())
	{
		return -1;
	}

	namedWindow(webcam, CV_WINDOW_AUTOSIZE);

	vector<Vec3d> rotationVector, translationVectors;

	while (1)
	{
		if (!vid.read(frame))
			break;

		aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
		aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rotationVector, translationVectors);

		for (int i = 0; i< markerIds.size(); i++)
		{
			aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVector[i], translationVectors[i], 0.1f);

		}

		imshow(webcam, frame);

		char character = waitKey(1000 / fps);


		switch (character)
		{
			case 27:
			//press escape key to exit
			return 1;
			break;
		}

	}
	return 1;
}

int main()
{
	/*
	**Note about the main function-
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
	*/
	const string webcam = "webcam";

	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

	Mat distanceCoefficients;
	
	//calibrateCameraCall(webcam,cameraMatrix,distanceCoefficients);
	//loadCameraCalibration("cameraCalibrated", cameraMatrix, distanceCoefficients);
	faceMasking(webcam, cameraMatrix, distanceCoefficients);
	//arucoDetection(webcam, cameraMatrix, distanceCoefficients);

	
	return 0;

}
