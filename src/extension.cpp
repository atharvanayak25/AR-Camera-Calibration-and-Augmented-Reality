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
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>

using namespace cv;
using namespace std;

// -----------------------------------------------------------------------------
// OBJ loader: loads vertices and faces from an OBJ file.
bool loadOBJ(const string& objFilePath, vector<Point3f>& outVertices, vector<Vec3i>& outFaces) {
    ifstream file(objFilePath);
    if (!file.is_open()) {
        cerr << "Error: Could not open OBJ file: " << objFilePath << endl;
        return false;
    }
    cout << "Loading OBJ file: " << objFilePath << endl;
    outVertices.clear();
    outFaces.clear();
    string line;
    while (getline(file, line)) {
        if (line.empty())
            continue;
        if (line.substr(0, 2) == "v ") { // Vertex line: "v x y z"
            istringstream iss(line);
            char vLabel;
            float x, y, z;
            if (!(iss >> vLabel >> x >> y >> z)) {
                cerr << "Error parsing vertex: " << line << endl;
                continue;
            }
            outVertices.push_back(Point3f(x, y, z));
        } else if (line.substr(0, 2) == "f ") { // Face line: "f idx/idx/idx ..."
            istringstream iss(line);
            char fLabel;
            iss >> fLabel; // skip 'f'
            vector<int> vertexIndices;
            string token;
            while (iss >> token) {
                istringstream tokenStream(token);
                string indexStr;
                if (getline(tokenStream, indexStr, '/')) {
                    try {
                        int index = stoi(indexStr);
                        vertexIndices.push_back(index - 1); // convert to 0-based
                    } catch (exception& e) {
                        cerr << "Error converting token to integer: " << token << endl;
                    }
                }
            }
            if (vertexIndices.size() == 3) {
                outFaces.push_back(Vec3i(vertexIndices[0], vertexIndices[1], vertexIndices[2]));
            } else if (vertexIndices.size() == 4) {
                outFaces.push_back(Vec3i(vertexIndices[0], vertexIndices[1], vertexIndices[2]));
                outFaces.push_back(Vec3i(vertexIndices[0], vertexIndices[2], vertexIndices[3]));
            } else {
                cerr << "Face with unsupported number of vertices: " << vertexIndices.size() << endl;
            }
        }
    }
    file.close();
    cout << "OBJ loaded: " << outVertices.size() << " vertices, " << outFaces.size() << " faces." << endl;
    return true;
}

// -----------------------------------------------------------------------------
// Adjusts (scales/translates) the model so it appears above the target.
void adjustModel(vector<Point3f>& vertices, float scale, float zOffset) {
    for (auto &v : vertices) {
        v.x *= scale;
        v.y *= scale;
        v.z *= scale;
        v.z += zOffset;
    }
}

// -----------------------------------------------------------------------------
// Orders 4 points into a consistent order: top-left, top-right, bottom-right, bottom-left.
vector<Point2f> orderPoints(vector<Point2f> pts) {
    vector<Point2f> ordered(4);
    float minSum = FLT_MAX, maxSum = -FLT_MAX;
    float minDiff = FLT_MAX, maxDiff = -FLT_MAX;
    int tlIdx = 0, brIdx = 0, trIdx = 0, blIdx = 0;
    for (int i = 0; i < 4; i++) {
        float sum = pts[i].x + pts[i].y;
        float diff = pts[i].x - pts[i].y;
        if (sum < minSum) { minSum = sum; tlIdx = i; }
        if (sum > maxSum) { maxSum = sum; brIdx = i; }
        if (diff < minDiff) { minDiff = diff; trIdx = i; }
        if (diff > maxDiff) { maxDiff = diff; blIdx = i; }
    }
    ordered[0] = pts[tlIdx];  // top-left
    ordered[1] = pts[trIdx];  // top-right
    ordered[2] = pts[brIdx];  // bottom-right
    ordered[3] = pts[blIdx];  // bottom-left
    return ordered;
}

// -----------------------------------------------------------------------------
// Detection: scans for a rectangular target using contour analysis.
// Returns true if a valid target is found and its ordered 4 corners.
bool detectTarget(const Mat &frame, vector<Point2f> &targetCorners) {
    Mat gray, blurred, edges;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 0);
    Canny(blurred, edges, 50, 150);
    
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    
    // Expected target dimensions: for example, an 8x6 rectangle (~1.33 aspect ratio)
    const double expectedRatio = 8.0 / 6.0;
    const double ratioTolerance = 0.5;
    const double minAreaThreshold = 1000.0;
    
    double bestArea = 0;
    vector<Point> bestContour;
    for (auto &contour : contours) {
        vector<Point> approx;
        double peri = arcLength(contour, true);
        approxPolyDP(contour, approx, 0.02 * peri, true);
        if (approx.size() == 4 && isContourConvex(approx)) {
            double area = contourArea(approx);
            if (area < minAreaThreshold)
                continue;
            RotatedRect rect = minAreaRect(approx);
            double width = rect.size.width, height = rect.size.height;
            if (height == 0)
                continue;
            double ratio = width / height;
            if (ratio < 1) ratio = 1.0 / ratio;
            if (fabs(ratio - expectedRatio) > ratioTolerance)
                continue;
            if (area > bestArea) {
                bestArea = area;
                bestContour = approx;
            }
        }
    }
    if (bestContour.empty() || bestContour.size() != 4)
        return false;
    
    vector<Point2f> corners;
    for (auto &pt : bestContour)
        corners.push_back(Point2f(pt.x, pt.y));
    
    targetCorners = orderPoints(corners);
    return true;
}

// -----------------------------------------------------------------------------
// Main: Uses a state machine that first detects a rectangle target, then
// tracks its corners using optical flow. If tracking fails, it reverts to detection.
int main() {
    // Load calibration parameters.
    FileStorage fs("../calibration/intrinsics.yaml", FileStorage::READ);
    if (!fs.isOpened()){
        cerr << "Error: Could not open calibration file ../calibration/intrinsics.yaml" << endl;
        return -1;
    }
    Mat cameraMatrix, distCoeffs;
    fs["CameraMatrix"] >> cameraMatrix;
    fs["DistortionCoefficients"] >> distCoeffs;
    fs.release();
    cout << "Loaded Camera Matrix:" << endl << cameraMatrix << endl;
    cout << "Loaded Distortion Coefficients:" << endl << distCoeffs.t() << endl;
    
    // Define the target's real-world coordinates (for an 8x6 rectangle).
    vector<Point3f> targetObjectPoints = {
        Point3f(0, 0, 0),    // top-left
        Point3f(8, 0, 0),    // top-right
        Point3f(8, 6, 0),    // bottom-right
        Point3f(0, 6, 0)     // bottom-left
    };
    
    // Load the OBJ model.
    string objFilePath = "../models/newcar.obj";
    vector<Point3f> objVertices;
    vector<Vec3i> objFaces;
    if (!loadOBJ(objFilePath, objVertices, objFaces))
        return -1;
    
    // Adjust the model so it appears above the target.
    adjustModel(objVertices, 1.0f, 5.0f);
    
    VideoCapture cap(0);
    if (!cap.isOpened()){
        cerr << "Error: Could not open the camera." << endl;
        return -1;
    }
    const string windowName = "AR Model with Detection & Tracking";
    
    // State variables for tracking.
    bool isTracking = false;
    vector<Point2f> targetCorners; // current corners (ordered)
    Mat prevFrame;  // previous frame for optical flow
    
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Captured empty frame." << endl;
            break;
        }
        
        // If not tracking, try to detect the target.
        if (!isTracking) {
            if (detectTarget(frame, targetCorners)) {
                isTracking = true;
                // Copy current frame for tracking reference.
                prevFrame = frame.clone();
                cout << "Target detected and locked." << endl;
            }
        } else {
            // Use optical flow to update the corner positions.
            vector<Point2f> newCorners;
            vector<uchar> status;
            vector<float> err;
            calcOpticalFlowPyrLK(prevFrame, frame, targetCorners, newCorners, status, err);
            
            // Check that all corners were successfully tracked.
            int goodCount = 0;
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i])
                    goodCount++;
            }
            // If too many points are lost, abandon tracking.
            if (goodCount < 4) {
                isTracking = false;
                targetCorners.clear();
                cout << "Lost tracking. Re-detecting target." << endl;
            } else {
                targetCorners = newCorners;
                prevFrame = frame.clone();
            }
        }
        
        // If we have a valid target, perform AR overlay.
        if (isTracking && targetCorners.size() == targetObjectPoints.size()) {
            // Draw the tracked target outline.
            for (int i = 0; i < 4; i++) {
                line(frame, targetCorners[i], targetCorners[(i+1)%4], Scalar(0, 255, 0), 2);
                circle(frame, targetCorners[i], 5, Scalar(0, 0, 255), -1);
            }
            
            // Estimate pose using solvePnP.
            Mat rvec, tvec;
            try {
                bool success = solvePnP(targetObjectPoints, targetCorners, cameraMatrix, distCoeffs, rvec, tvec);
                if (success) {
                    drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 3);
                    
                    // Project and render the OBJ model.
                    vector<Point2f> projectedPoints;
                    projectPoints(objVertices, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
                    if (projectedPoints.size() == objVertices.size()) {
                        for (size_t i = 0; i < objFaces.size(); i++) {
                            int i1 = objFaces[i][0];
                            int i2 = objFaces[i][1];
                            int i3 = objFaces[i][2];
                            if(i1 < 0 || i1 >= projectedPoints.size() ||
                               i2 < 0 || i2 >= projectedPoints.size() ||
                               i3 < 0 || i3 >= projectedPoints.size())
                                continue;
                            line(frame, projectedPoints[i1], projectedPoints[i2], Scalar(255, 255, 255), 2);
                            line(frame, projectedPoints[i2], projectedPoints[i3], Scalar(255, 255, 255), 2);
                            line(frame, projectedPoints[i3], projectedPoints[i1], Scalar(255, 255, 255), 2);
                        }
                    } else {
                        cerr << "Mismatch in projected points and model vertices." << endl;
                    }
                } else {
                    cout << "Pose estimation failed." << endl;
                }
            } catch (const Exception &e) {
                cerr << "Exception in solvePnP: " << e.what() << endl;
            }
        } else {
            putText(frame, "Target not detected", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 2);
        }
        
        imshow(windowName, frame);
        char key = (char)waitKey(10);
        if (key == 27) // ESC to exit
            break;
    }
    
    cap.release();
    destroyAllWindows();
    return 0;
}