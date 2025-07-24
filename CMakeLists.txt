# Akshaj Raut
# Atharva Nayak

# CS 5330 Computer Vision
# Spring 2025

# Project 4 - Calibration and Augmented Reality

cmake_minimum_required(VERSION 3.10)
project(OpenCVProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV and set include directories
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})

add_executable(pose pose.cpp)
target_link_libraries(pose ${OpenCV_LIBS})

add_executable(readobj read_obj.cpp)
target_link_libraries(readobj ${OpenCV_LIBS})

add_executable(orb orb.cpp)
target_link_libraries(orb ${OpenCV_LIBS})

add_executable(extension extension.cpp)
target_link_libraries(extension ${OpenCV_LIBS})
