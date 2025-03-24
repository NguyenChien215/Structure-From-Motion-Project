import cv2
import numpy as np
import os
from Config import CALIBRATION_CHESSBOARD_FOLDER

# Prepare object points
chessboard_size = (11, 7) # 12x8 chessboard size
square_size = 20
scale_factor = 0.5

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load all images from the folder
folder_path = CALIBRATION_CHESSBOARD_FOLDER
images = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.lower().endswith(('.jpg', '.png'))]

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners 
        img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)

        # Resize the image before displaying 
        height, width = img.shape[:2]
        scaled_img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))

        # Show the scaled image
        cv2.imshow('Chessboard', scaled_img)
        cv2.waitKey(10)
    else:
        print(f"Chessboard not found in {fname}")

cv2.destroyAllWindows()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save K and dist to a file
with open("output/custom_image/Intrinsic_parameter_K.txt", "w") as f:
    f.write("Camera matrix (K):\n")
    np.savetxt(f, mtx, fmt="%.6f")

    f.write("\nDistortion coefficients (dist):\n")
    np.savetxt(f, dist, fmt="%.6f")

# Print the results
print("Camera matrix:")
print(mtx)
print("\nDistortion coefficients:")
print(dist)

