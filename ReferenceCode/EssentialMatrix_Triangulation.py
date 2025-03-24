import cv2
import numpy as np
import matplotlib.pyplot as plt
from Config import GIVEN_DATA_IMG1, GIVEN_DATA_IMG2, K, RATIO_COEFFICIENT

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

# Compute the essential matrix (E = K^T * F * K)
E = K.T @ F @ K

# Decompose the essential matrix using SVD
U, _, Vt = np.linalg.svd(E)
W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# Possible rotation and translation solutions
R1 = U @ W @ Vt
R2 = U @ W.T @ Vt
t = U[:, 2]

solutions = [
    (R1, t),
    (R1, -t),
    (R2, t),
    (R2, -t)
]

# Camera matrix for P1 (first camera is the identity matrix)
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P1 = K @ P1

best_points = None
best_solution = None

# Try each solution and triangulate points
for idx, (R, t) in enumerate(solutions, 1):
    P2 = np.hstack((R, t.reshape(-1, 1)))
    P2 = K @ P2

    # Triangulate points
    pts1_h = cv2.convertPointsToHomogeneous(pts1)[:, 0, :2].T
    pts2_h = cv2.convertPointsToHomogeneous(pts2)[:, 0, :2].T
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]
    points_3d = points_3d.T
    
    # Count the number of points with positive depth
    num_positive_depths = np.sum(points_3d[:, 2] > 0)

    # Print the number of positive depth points for each solution
    print(f"Solution {idx}: {num_positive_depths} points with positive depth.")

    # Choose the solution with the most points in front of both cameras
    if best_points is None or num_positive_depths > np.sum(best_points[:, 2] > 0):
        best_points = points_3d
        best_solution = (R, t)

# Print the best rotation and translation
print("\nBest Rotation:\n", best_solution[0])
print("Best Translation:\n", best_solution[1])

# Show 3D points before bundle adjustment
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(best_points[:, 0], best_points[:, 1], best_points[:, 2], s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Adjust the view for better visibility
ax.view_init(azim=-60, elev=30)
plt.show()
