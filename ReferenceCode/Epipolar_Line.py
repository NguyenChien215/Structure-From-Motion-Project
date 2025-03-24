import cv2
import numpy as np
import matplotlib.pyplot as plt
from Config import GIVEN_DATA_IMG1, GIVEN_DATA_IMG2, RATIO_COEFFICIENT

# Load the images
img1 = cv2.imread(GIVEN_DATA_IMG1)
img2 = cv2.imread(GIVEN_DATA_IMG2)

if img1 is None or img2 is None:
    raise FileNotFoundError("Please make sure the image filenames are correct!")

# Initialize SIFT detector
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Match the descriptors using BFMatcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test to get good matches
good_matches = []
for m, n in matches:
    if m.distance < RATIO_COEFFICIENT * n.distance:
        good_matches.append(m)

# Get the matching points
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

# Compute the fundamental matrix
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

# Draw the interest points on the first image
img1_with_points = img1.copy()
for pt in pts1:
    cv2.circle(img1_with_points, tuple(pt.astype(int)), 5, (0, 255, 0), -1)

# Compute epipolar lines for the points in the first image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)

# Function to draw the epipolar lines on the second image
r, c, _ = img2.shape
img2_with_lines = img2.copy()
for r in lines2:
    color = tuple(np.random.randint(0, 255, 3).tolist())
    x0, y0 = map(int, [0, -r[2] / r[1]])
    x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
    img2_with_lines = cv2.line(img2_with_lines, (x0, y0), (x1, y1), color, 1)

# Display the images
plt.figure(figsize=(10, 5))

# Display the first image with interest points
plt.subplot(121)
plt.imshow(cv2.cvtColor(img1_with_points, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Interest Points (Image 1)')

# Display the second image with epipolar lines
plt.subplot(122)
plt.imshow(cv2.cvtColor(img2_with_lines, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Epipolar Lines (Image 2)')

plt.show()
