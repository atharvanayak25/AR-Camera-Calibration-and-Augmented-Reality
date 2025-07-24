/*
Akshaj Raut
Atharva Nayak

CS 5330 Computer Vision
Spring 2025

Project 4 - Calibration and Augmented Reality
*/

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // Open the default camera.
    VideoCapture cap(0);
    if (!cap.isOpened()){
        cerr << "Error: Could not open the camera." << endl;
        return -1;
    }
    
    // Create an ORB feature detector.
    // You can adjust parameters such as the number of features or FAST threshold.
    Ptr<ORB> orb = ORB::create(
        500,    // maximum number of features to retain
        1.2f,   // scale factor between pyramid levels
        8,      // number of pyramid levels
        31,     // edge threshold size (affects feature detection near image boundaries)
        0,      // first level
        2,      // WTA_K (number of points that produce each element of the oriented BRIEF descriptor)
        ORB::HARRIS_SCORE, // score type: HARRIS_SCORE or FAST_SCORE
        31,     // patch size
        20      // fastThreshold: higher value means fewer detected features
    );
    
    // Create a window for display.
    const string windowName = "ORB Feature Detection";
    namedWindow(windowName, WINDOW_AUTOSIZE);
    
    while (true)
    {
        Mat frame, gray;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Captured empty frame." << endl;
            break;
        }
        
        // Convert to grayscale.
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // Detect ORB keypoints and compute descriptors.
        vector<KeyPoint> keypoints;
        Mat descriptors;
        orb->detectAndCompute(gray, Mat(), keypoints, descriptors);
        
        // Draw keypoints on the original frame.
        Mat output;
        drawKeypoints(frame, keypoints, output, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        imshow(windowName, output);
        char key = (char)waitKey(30);
        if (key == 27) // ESC to exit
            break;
    }
    
    cap.release();
    destroyAllWindows();
    return 0;
}