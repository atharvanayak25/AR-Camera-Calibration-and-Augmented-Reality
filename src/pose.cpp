/*
Akshaj Raut
Atharva Nayak

CS 5330 Computer Vision
Spring 2025

Project 4 - Calibration and Augmented Reality
*/

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <utility>

using namespace cv;
using namespace std;

int main()
{
    // Load calibration parameters from file (using .yaml extension)
    FileStorage fs("../calibration/intrinsics.yaml", FileStorage::READ);
    if (!fs.isOpened()){
        cerr << "Error: Could not open calibration file ../calibration/intrinsics.yaml" << endl;
        return -1;
    }
    Mat cameraMatrix, distCoeffs;
    fs["CameraMatrix"] >> cameraMatrix;
    fs["DistortionCoefficients"] >> distCoeffs;
    fs.release();

    // Define checkerboard pattern size (internal corners: 9 columns, 6 rows)
    Size patternSize(9, 6);

    // Prepare 3D world coordinates for the entire checkerboard.
    // The board lies in the plane z=0 with the top-left corner at (0,0,0).
    vector<Point3f> boardObjectPoints;
    for (int i = 0; i < patternSize.height; i++){
        for (int j = 0; j < patternSize.width; j++){
            boardObjectPoints.push_back(Point3f(j, -i, 0));
        }
    }

    // Define a virtual pyramid object.
    // The base is a square and the apex is above the base.
    // Base: a square of size 2 units, placed at z = 5.0
    // Apex: centered above the base at (1, -1) with z = 8.0.
    vector<Point3f> pyramidPoints;
    // Base vertices
    pyramidPoints.push_back(Point3f(0, 0, 5.0));    // Vertex 0: bottom-left
    pyramidPoints.push_back(Point3f(2, 0, 5.0));    // Vertex 1: bottom-right
    pyramidPoints.push_back(Point3f(2, -2, 5.0));   // Vertex 2: top-right
    pyramidPoints.push_back(Point3f(0, -2, 5.0));   // Vertex 3: top-left
    // Apex
    Point3f apex(1, -1, 8.0);                        // Vertex 4: apex
    pyramidPoints.push_back(apex);

    // Define the edges of the pyramid:
    // Base outline and edges connecting base vertices to the apex.
    vector<pair<int, int>> pyramidEdges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Base outline
        {0, 4}, {1, 4}, {2, 4}, {3, 4}  // Side edges to apex
    };

    // Open the default camera
    VideoCapture cap(0);
    if (!cap.isOpened()){
        cerr << "Error: Could not open the camera." << endl;
        return -1;
    }
    const string windowName = "Camera Pose & Virtual Object (Pyramid)";

    while (true)
    {
        Mat frame, gray;
        cap >> frame;
        if (frame.empty()){
            cerr << "Error: Captured empty frame." << endl;
            break;
        }
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect the checkerboard corners using a fast check to improve performance
        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, patternSize, corners,
                                           CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
        if(found)
        {
            // Refine corner locations for increased accuracy
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            drawChessboardCorners(frame, patternSize, Mat(corners), found);

            // Estimate the camera pose using solvePnP.
            Mat rvec, tvec;
            bool success = solvePnP(boardObjectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);
            if(success)
            {
                // Draw coordinate axes on the board (axis length = 3 units)
                drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 3);

                // Project the pyramid's 3D points into the image plane.
                vector<Point2f> imagePoints;
                projectPoints(pyramidPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

                // Draw the pyramid edges by connecting the projected points.
                for (size_t i = 0; i < pyramidEdges.size(); i++){
                    Point pt1 = imagePoints[pyramidEdges[i].first];
                    Point pt2 = imagePoints[pyramidEdges[i].second];
                    line(frame, pt1, pt2, Scalar(255, 0, 0), 2);  // Blue lines for the pyramid
                }
            }
            else
            {
                cout << "Pose estimation failed." << endl;
            }
        }
        // Display the frame
        imshow(windowName, frame);
        char key = (char)waitKey(10);
        if(key == 27) // ESC key to exit
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}