import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables from the .env file (If don't have .env file it will choose the default value)
load_dotenv()

def load_camera_matrix(filename="output/custom_image/Intrinsic_parameter_K.txt"):
    with open(filename, "r") as f:
        lines = f.readlines()

    # Find the position of Camera matrix (K)
    k_start = lines.index("Camera matrix (K):\n") + 1
    k_end = k_start + 3  

    # Find the position of Distortion coefficients 
    dist_start = lines.index("Distortion coefficients (dist):\n") + 1

    # Read K matrix
    my_K = np.loadtxt(lines[k_start:k_end])

    # Read Distortion coefficients
    my_dist = np.loadtxt(lines[dist_start:])

    return my_K, my_dist


# Get the environment variables with default values if not found
GIVEN_DATA_IMG1 = os.getenv("GIVEN_DATA_IMG1", "my_data/MESONA1.JPG")
GIVEN_DATA_IMG2 = os.getenv("GIVEN_DATA_IMG2", "my_data/MESONA2.JPG")
MY_DATA_IMG1 = os.getenv("MY_DATA_IMG1", "my_data/custom_thing1.jpg")
MY_DATA_IMG2 = os.getenv("MY_DATA_IMG2", "my_data/custom_thing2.jpg")
CALIBRATION_CHESSBOARD_FOLDER = os.getenv("CALIBRATION_CHESSBOARD_FOLDER", "Chessboard")  
DAVID_LOWE_COEFFICIENT = os.getenv("DAVID_LOWE_COEFFICIENT", 0.75)

# Load K from environment variable with the value calculated from the calibration process
K_str = os.getenv("K", "[ [1.4219, 0.0005, 0.5092], [0, 1.4219, 0], [0, 0, 0.001] ]")
K = np.array(eval(K_str))

# Load the custom K
my_K, my_dist = load_camera_matrix()
print("Loaded Camera Matrix K:\n", my_K)
print("Loaded Distortion Coefficients:\n", my_dist)


