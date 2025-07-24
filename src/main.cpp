/*
Akshaj Raut
Atharva Nayak

CS 5330 Computer Vision
Spring 2025

Project 4 - Calibration and Augmented Reality
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
    // Open the default camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open the camera." << endl;
        return -1;
    }

    // Define the checkerboard pattern size (internal corners: 9 columns, 6 rows)
    Size patternSize(9, 6);
    const string windowName = "Checkerboard Calibration";

    // Scale factor for processing (process a downscaled version for speed)
    double scaleFactor = 0.5;

    int frameCount = 0;
    
    // Containers for calibration data
    vector<vector<Point2f>> corner_list;       // 2D image points for each saved image
    vector<vector<Vec3f>> point_list;            // Corresponding 3D world points for each saved image
    vector<Mat> image_list;                      // Calibration images

    // Variables to store the last valid detection (image and corners)
    vector<Point2f> lastValidCorners;
    Mat lastValidImage;

    // Variables for calibration results
    bool calibrated = false;
    Mat cameraMatrix, distCoeffs;
    double reprojectionError = 0.0;
    vector<Mat> rvecs, tvecs;

    cout << "Press 's' to save a calibration frame, 'c' to calibrate (min 5 frames), " 
         << "and 'w' to write intrinsic parameters to file." << endl;

    while (true)
    {
        Mat fullFrame;
        cap >> fullFrame;
        if (fullFrame.empty()) {
            cerr << "Error: Captured empty frame." << endl;
            break;
        }

        // Downscale the full resolution frame for faster processing
        Mat smallFrame, smallGray;
        resize(fullFrame, smallFrame, Size(), scaleFactor, scaleFactor, INTER_LINEAR);
        cvtColor(smallFrame, smallGray, COLOR_BGR2GRAY);

        // Detect the checkerboard corners on the downscaled frame
        vector<Point2f> cornerSet;
        bool patternFound = findChessboardCorners(smallGray, patternSize, cornerSet,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);

        if (patternFound) {
            // Refine corner locations
            cornerSubPix(smallGray, cornerSet, Size(11, 11), Size(-1, -1),
                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

            // Scale detected corner coordinates back to full resolution
            for (auto &pt : cornerSet) {
                pt.x /= scaleFactor;
                pt.y /= scaleFactor;
            }

            // Draw the detected corners on the full resolution frame
            drawChessboardCorners(fullFrame, patternSize, Mat(cornerSet), patternFound);

            // Update the last valid detection
            lastValidCorners = cornerSet;
            lastValidImage = fullFrame.clone();
        }

        // Display instructions on the frame
        putText(fullFrame, "Press 's' to save frame, 'c' to calibrate, 'w' to write params", 
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);

        // Show the full-resolution frame with drawn corners
        imshow(windowName, fullFrame);

        // Wait for key press (1ms delay)
        char key = (char)waitKey(1);
        if (key == 27) { // ESC key exits
            break;
        }
        else if (key == 's' || key == 'S') {
            // Save calibration data only if we have a valid detection
            if (!lastValidCorners.empty()) {
                // Save the detected 2D corners
                corner_list.push_back(lastValidCorners);
                
                // Generate the 3D world point set corresponding to the checkerboard corners.
                // Assuming each square is 1 unit in size.
                vector<Vec3f> point_set;
                for (int i = 0; i < patternSize.height; i++) {
                    for (int j = 0; j < patternSize.width; j++) {
                        // (0,0,0) is the upper-left corner; next row has y coordinate -1.
                        point_set.push_back(Vec3f(j, -i, 0));
                    }
                }
                point_list.push_back(point_set);
                
                // Save the calibration image used for this detection
                image_list.push_back(lastValidImage);
                cout << "Calibration frame saved. Total frames: " << corner_list.size() << endl;
            } else {
                cout << "No valid detection available to save." << endl;
            }
        }
        else if (key == 'c' || key == 'C') {
            // Run calibration if enough frames have been collected (minimum 5)
            if (corner_list.size() >= 5) {
                // Use the size of the first saved calibration image
                Size imageSize = image_list[0].size();

                // Initialize the camera matrix as an identity matrix and set the center
                cameraMatrix = Mat::eye(3, 3, CV_64F);
                cameraMatrix.at<double>(0, 0) = 1.0;
                cameraMatrix.at<double>(0, 2) = imageSize.width / 2.0;
                cameraMatrix.at<double>(1, 1) = 1.0;
                cameraMatrix.at<double>(1, 2) = imageSize.height / 2.0;

                // Initialize distortion coefficients (5 parameters)
                distCoeffs = Mat::zeros(5, 1, CV_64F);

                // Use CALIB_FIX_ASPECT_RATIO flag so that the two focal lengths are the same
                int flags = CALIB_FIX_ASPECT_RATIO;
                reprojectionError = calibrateCamera(point_list, corner_list, imageSize,
                                                    cameraMatrix, distCoeffs, rvecs, tvecs, flags);
                cout << "Calibration complete." << endl;
                cout << "Camera Matrix:\n" << cameraMatrix << endl;
                cout << "Distortion Coefficients:\n" << distCoeffs.t() << endl;
                cout << "Reprojection Error: " << reprojectionError << " pixels" << endl;
                calibrated = true;
            } else {
                cout << "Need at least 5 calibration images. Currently: " << corner_list.size() << endl;
            }
        }
        else if (key == 'w' || key == 'W') {
            // Write the intrinsic parameters to a file if calibration has been done.
            if (calibrated) {
                FileStorage fs("../calibration/intrinsics.yaml", FileStorage::WRITE);
                if (!fs.isOpened()) {
                    cerr << "Error: Could not open file for writing." << endl;
                } else {
                    fs << "CameraMatrix" << cameraMatrix;
                    fs << "DistortionCoefficients" << distCoeffs;
                    fs << "ReprojectionError" << reprojectionError;
                    fs.release();
                    cout << "Calibration parameters saved to ../calibration/intrinsics.yaml" << endl;
                }
            } else {
                cout << "Camera not calibrated yet. Press 'c' to calibrate." << endl;
            }
        }
        frameCount++;
    }

    // Optionally, save the calibration images to disk in the ../calibration/ folder
    for (size_t i = 0; i < image_list.size(); i++) {
        string filename = "../calibration/calibration_image_" + to_string(i) + ".png";
        imwrite(filename, image_list[i]);
    }
    cout << "Total calibration images saved to disk: " << image_list.size() << endl;

    // Release resources and close windows
    cap.release();
    destroyAllWindows();

    return 0;
}