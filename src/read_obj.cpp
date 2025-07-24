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
#include <utility>

using namespace cv;
using namespace std;

// Revised OBJ loader for vertices (v) and faces (f)
// Handles face tokens with slashes and supports triangles and quadrilaterals.
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

        // Vertex line: "v x y z"
        if (line.substr(0, 2) == "v ") {
            istringstream iss(line);
            char vLabel;
            float x, y, z;
            if (!(iss >> vLabel >> x >> y >> z)) {
                cerr << "Error parsing vertex: " << line << endl;
                continue;
            }
            outVertices.push_back(Point3f(x, y, z));
        }
        // Face line: can be triangle or quadrilateral with tokens like "1/1/1"
        else if (line.substr(0, 2) == "f ") {
            istringstream iss(line);
            char fLabel;
            iss >> fLabel; // Skip the 'f'
            vector<int> vertexIndices;
            string token;
            while (iss >> token) {
                // Token might be "1/1/1" or similar.
                istringstream tokenStream(token);
                string indexStr;
                if (getline(tokenStream, indexStr, '/')) {
                    try {
                        int index = stoi(indexStr);
                        vertexIndices.push_back(index - 1); // Convert to 0-based index.
                    } catch (exception& e) {
                        cerr << "Error converting token to integer: " << token << endl;
                    }
                }
            }
            // Triangulate: if three indices, use one triangle; if four, split into two triangles.
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

// Optional function to scale/translate the model so it appears above the board.
void adjustModel(vector<Point3f>& vertices, float scale, float zOffset) {
    for (auto &v : vertices) {
        v.x *= scale;
        v.y *= scale;
        v.z *= scale;
        v.z += zOffset;
    }
}

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

    cout << "Loaded Camera Matrix:" << endl << cameraMatrix << endl;
    cout << "Loaded Distortion Coefficients:" << endl << distCoeffs.t() << endl;

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

    // Load the OBJ model.
    string objFilePath = "../models/newcar.obj";
    vector<Point3f> objVertices;
    vector<Vec3i> objFaces;
    if (!loadOBJ(objFilePath, objVertices, objFaces)) {
        return -1;
    }

    // Optionally adjust the model so it appears above the board.
    // For instance, scale by 1.0 and translate up by 5.0 units in z.
    adjustModel(objVertices, 1.0f, 5.0f);

    // Open the default camera.
    VideoCapture cap(0);
    if (!cap.isOpened()){
        cerr << "Error: Could not open the camera." << endl;
        return -1;
    }
    const string windowName = "OBJ Model AR";
    
    while (true)
    {
        Mat frame, gray;
        cap >> frame;
        if (frame.empty()){
            cerr << "Error: Captured empty frame." << endl;
            break;
        }
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect the checkerboard corners using a fast check.
        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, patternSize, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
        if(found && (int)corners.size() == patternSize.area())
        {
            // Refine the corner locations.
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            drawChessboardCorners(frame, patternSize, Mat(corners), found);

            // Estimate the camera pose using solvePnP.
            Mat rvec, tvec;
            bool success = solvePnP(boardObjectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);
            if(success)
            {
                // Draw coordinate axes on the board (axis length = 3 units).
                drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 3);

                // Project the OBJ model vertices into the image plane.
                vector<Point2f> projectedPoints;
                projectPoints(objVertices, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

                // Verify that the projected points vector size matches the number of vertices.
                if(projectedPoints.size() != objVertices.size()){
                    cerr << "Error: projectedPoints size (" << projectedPoints.size()
                         << ") does not match objVertices size (" << objVertices.size() << ")." << endl;
                    continue;
                }

                // Draw the OBJ model in a wireframe style using the triangular faces.
                for (size_t i = 0; i < objFaces.size(); i++){
                    int i1 = objFaces[i][0];
                    int i2 = objFaces[i][1];
                    int i3 = objFaces[i][2];

                    // Verify that indices are within bounds.
                    if(i1 < 0 || i1 >= projectedPoints.size() ||
                       i2 < 0 || i2 >= projectedPoints.size() ||
                       i3 < 0 || i3 >= projectedPoints.size()){
                        cerr << "Face " << i << " has invalid indices: " 
                             << i1 << ", " << i2 << ", " << i3 << endl;
                        continue;
                    }

                    Point pt1 = projectedPoints[i1];
                    Point pt2 = projectedPoints[i2];
                    Point pt3 = projectedPoints[i3];

                    // Draw triangle edges for a wireframe look.
                    line(frame, pt1, pt2, Scalar(255, 255, 255), 2);
                    line(frame, pt2, pt3, Scalar(255, 255, 255), 2);
                    line(frame, pt3, pt1, Scalar(255, 255, 255), 2);
                }
            }
            else {
                cout << "Pose estimation failed." << endl;
            }
        }

        imshow(windowName, frame);
        char key = (char)waitKey(10);
        if(key == 27) // ESC key to exit
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}