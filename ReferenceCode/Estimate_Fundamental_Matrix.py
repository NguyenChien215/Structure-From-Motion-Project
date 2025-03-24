import cv2
import numpy as np
import matplotlib.pyplot as plt
from Config import GIVEN_DATA_IMG1, GIVEN_DATA_IMG2, RATIO_COEFFICIENT

# Read the two images
img1 = cv2.imread(GIVEN_DATA_IMG1)
img2 = cv2.imread(GIVEN_DATA_IMG2)

if img1 is None or img2 is None:
    raise FileNotFoundError("Please make sure the image filenames are correct!")

# Use SIFT to detect and describe keypoints
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Match descriptors using BFMatcher with the ratio test (Lowe's ratio test)
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Filter the matches using the ratio test
good_matches = []
for m, n in matches:
    if m.distance < RATIO_COEFFICIENT * n.distance:
        good_matches.append(m)

# Get the corresponding points for the good matches
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

# Compute the fundamental matrix using the 8-point algorithm with normalization
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

# Print the fundamental matrix
print("Fundamental Matrix (F):\n", F)
