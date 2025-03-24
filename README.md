## Author
Name: Nguyen Quang Chien
ID: 2431101
Email: 2431101@hcmute.edu.vn
Date: 23/03/2025

## Structure from motion project
This project implements a Structure From Motion (SFM) algorithm to reconstruct a 3D structure from 2D images.

## Objective
+ Reconstruct a 3D structure from a set of 2D images
+ Detect and extract interesting points (keypoints) using feature detection techniques (SIFT).
+ Perform feature matching across images to establish correspondences between keypoints. (Homography, RANSAC)
+ Compute the fundamental matrix to estimate the epipolar geometry between image pairs.
+ Use Structure from Motion (SFM) to estimate camera positions and orientations.

## Structure of the repositories
Project Root 
│── Chessboard/
│── my_data/
│── output/
│── ReferenceCode/
│── .python-version
│── 1_CameraLib.py
│── 2_Structure_From_Motion_Given_Image.py
│── 3_Structure_From_Motion_Custom_Image.py
│── Config.py
│── pyproject.toml
└── README.md

+ Chessboard: Folder contains the images of the chessboard for camera calibration
+ my_data: The input data include given data and custom data, this data is use for structure from motion
+ output: The output data when running SfM project (include the Extracting feature, feature matching, camera parameter, ...)
+ Reference code: The reference code from old lesson 
+ 1_CameraLib.py: Python script running for find the Intrinsic_parameter_K of the camera
+ 2_Structure_From_Motion_Given_Image.py: python script running for structure from motion using given data
+ 3_Structure_From_Motion_Custom_Image.py: python script running for structure from motion using custom data
+ Config.py: Config environment and define some parameter, need to check the config.py first for change the parameter for suitable purpose
+ pyproject.toml: Create the environment using uv package (list all the dependencies need to use, equal the requirements.txt file)
+ README.md: Introduction and guide to run this project

## Step 1: Install the python and pip in the system (Can skip this step if python and pip is installed)
+ Download in the website: <https://www.python.org/downloads/> (Recommend python >= 3.12.8)
+ When install python please sure choose: "Add Python to PATH" option
+ Add in the path (If forget to choose "Add Python to PATH" please follow this video: <https://www.youtube.com/watch?v=91SGaK7_eeY>)
+ Check python by open the terminal and type "python --version" (If show out the version if python is installed correctly)
+ Check pip by open the terminal and type "pip --version" (if not have pip you can download and run this python script: <https://bootstrap.pypa.io/get-pip.py>)

## Step 2: Install uv for managing the packages
+ Go to the current directory (Directory which have pyproject.toml file)
+ Run "pip install uv" in the terminal
+ Run "uv --version" for validated uv
+ run "uv sync" (Make sure the pyproject.toml is in your current directory)

## Step 3: Running the project
+ First you should check the Config.py first for choose your data you want to run, can change the data if need.
+ Run command "uv run 1_CameraLib.py" for take the K parameter
+ Run command "uv run 2_Stucture_From_Motion_Given_Image.py" to run project using given image
+ Run command "uv run 3_3_Stucture_From_Motion_Custom_Image.py" to run project using custom image
+ After run you can check the result in the output/ folder


