import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from Config import GIVEN_DATA_IMG1, GIVEN_DATA_IMG2, DAVID_LOWE_COEFFICIENT, K

def draw_epipolar_lines_with_keypoints(img1, img2, F, pts1, pts2):
    """Draw keypoints on img1 and epipolar lines of those points on img2."""
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Compute epipolar lines in img2 corresponding to points in img1
    lines = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    # Draw keypoints on img1
    for pt in pts1:
        cv2.circle(img1_color, tuple(pt.ravel().astype(int)), 5, (0, 255, 0), -1)  # Green keypoints
    
    # Draw epipolar lines on img2
    for r, pt in zip(lines, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())  # Random color
        x0, y0 = map(int, [0, -r[2] / r[1]])  # Intersection with left border
        x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]])  # Right border
        cv2.line(img2_color, (x0, y0), (x1, y1), color, 1)

    return img1_color, img2_color

# Load the images
img1 = cv2.imread(GIVEN_DATA_IMG1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(GIVEN_DATA_IMG2, cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Draw keypoints
img1_store = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_store = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img1_kp = cv2.drawKeypoints(img1, kp1, None, color = (0, 255, 0))
img2_kp = cv2.drawKeypoints(img2, kp2, None, color = (0, 255, 0))

# Save the images
cv2.imwrite("output/given_image/SIFT_Detector_Image1.png", img1_store)
cv2.imwrite("output/given_image/SIFT_Detector_Image2.png", img2_store)

# Create a BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Match descriptors
matches = bf.knnMatch(des1, des2, k=2)
bf_matches = [m for m, n in matches]
print("Number matches when apply only BFMatcher:", len(bf_matches)) 

# Draw matches
bf_draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
bf_result = cv2.drawMatches(img1, kp1, img2, kp2, bf_matches, None, **bf_draw_params)

# Store images
cv2.imwrite("output/given_image/Matching_TwoImages_OnlyBFM_AllMatches.png", bf_result)

# Draw matches again with 1000 matches
bf_result = cv2.drawMatches(img1, kp1, img2, kp2, bf_matches[:1000], None, **bf_draw_params)
cv2.imwrite(f"output/given_image/Matching_TwoImages_OnlyBFM_First1000Matches.png", bf_result)

# Filter matches with Lowe’s ratio test
good_matches = []
for m, n in matches:
    if m.distance < DAVID_LOWE_COEFFICIENT * n.distance:
        good_matches.append(m)
print("Number matches when after use Lowe's ratio test:", len(good_matches)) 

# Draw matches & store images
Low_draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
Low_result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **Low_draw_params)
cv2.imwrite(f"output/given_image/Matching_TwoImages_AfterLowRatio_{len(good_matches)}Matches.png", bf_result)

# Apply RANSAC to find homography
if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is not None and mask is not None:
        matchesMask = mask.ravel().tolist()
        
        # Keep only inliers
        inlier_matches = [m for m, inlier in zip(good_matches, matchesMask) if inlier]

        num_inliers = np.sum(mask)  # Count the number of inliers
        print(f"Number of inliers: {num_inliers}/{len(good_matches)} ({(num_inliers / len(good_matches)) * 100:.2f}%)")
    else:
        matchesMask = None
        inlier_matches = []
        print("Homography computation failed.")
else:
    matchesMask = None
    inlier_matches = []
    print("Not enough matches for Homography.")

# Draw matches & store images
homo_draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
homo_result = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, **homo_draw_params)
cv2.imwrite(f"output/given_image/Matching_TwoImages_HomoRANSAC_{len(inlier_matches)}Matches.png", homo_result)

# Calculate the fundamental matrix
if len(inlier_matches) > 8: 
    src_pts = np.float32([kp1[m.queryIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 5.0, 0.99)

    with open("output/given_image/Fundamental_matrix.txt", "w") as f:
        f.write("Fundamental matrix (F):\n")
        np.savetxt(f, F, fmt="%.6f")
        print("Fundamental Matrix:\n", F)

    # Extract inlier points
    inlier_pts1 = src_pts[mask.ravel() == 1]
    inlier_pts2 = dst_pts[mask.ravel() == 1]

    # Draw keypoints on img1 & epipolar lines on img2
    kp_epipolar1, epipolar2 = draw_epipolar_lines_with_keypoints(
        img1, img2, F, inlier_pts1, inlier_pts2)

    # Draw keypoints on img2 & epipolar lines on img1
    kp_epipolar2, epipolar1 = draw_epipolar_lines_with_keypoints(
        img2, img1, F, inlier_pts2, inlier_pts1)

    # Create a side-by-side visualization
    combined = np.hstack((kp_epipolar1, epipolar2))
    cv2.imwrite("output/given_image/Combined_kp1_epipolar2.png", combined)
    combined = np.hstack((kp_epipolar2, epipolar1))
    cv2.imwrite("output/given_image/Combined_kp2_epipolar1.png", combined)
    
else:
    print("Not enough matches to compute Fundamental Matrix")

# Calculated Essential Matrix
E = K.T @ F @ K
with open("output/given_image/Essential_matrix.txt", "w") as f:
    f.write("Essential matrix (E):\n")
    np.savetxt(f, E, fmt="%.6f")
    print("Essential Matrix:\n", E)

# Decompose Essential Matrix to R and T
R1, R2, T = cv2.decomposeEssentialMat(E)

# Four solutions from Essential Matrix
possible_poses = [(R1, T), (R1, -T), (R2, T), (R2, -T)]

# Choose the best pose
best_pose = None
best_points = None

# Camera 1 Projection Matrix: [I | 0]
# This is the default camera at the origin with no rotation or translation.
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  

# Iterate through possible camera poses (R, T) for Camera 2
for i, (R, T) in enumerate(possible_poses):
    # Construct the projection matrix for Camera 2: [R | T]
    P2 = np.hstack((R, T))  

    # Convert 2D points to homogeneous coordinates by undistorting them
    # This removes lens distortion effects using the intrinsic matrix (K)
    pts1_h = cv2.undistortPoints(inlier_pts1.reshape(-1, 1, 2), K, None)
    pts2_h = cv2.undistortPoints(inlier_pts2.reshape(-1, 1, 2), K, None)

    # Perform triangulation to obtain 3D homogeneous points
    points4D = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)

    # Convert homogeneous coordinates (x, y, z, w) to 3D (x/w, y/w, z/w)
    points3D = points4D[:3] / points4D[3]

    # Count the number of points that have positive Z coordinates (in front of the camera)
    num_valid = np.sum(points3D[2] > 0)

    # Print the number of valid points for this pose
    print(f"Pose {i}: num_valid = {num_valid}")

    # Select the best pose that results in the highest number of valid 3D points
    if best_pose is None or num_valid > np.sum(best_points[2] > 0):
        best_pose = (R, T)
        best_points = points3D

# Calculated Essential Matrix
with open("output/given_image/Essential_matrix.txt", "a") as f:
    f.write("\nRotation Matrix (R):\n")
    np.savetxt(f, best_pose[0], fmt="%.6f")
    f.write("\nTranslation Matrix (T):\n")
    np.savetxt(f, best_pose[1], fmt="%.6f")
    print("\nRotation Matrix: \n", best_pose[0])
    print("\nTranslation Matrix: \n", best_pose[1])

# Store the 3D points
np.savetxt("output/given_image/3D_points.txt", best_points.T, fmt="%.6f")
print("3D points saved in '3D_points.txt'")

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Scatter plot for reconstructed 3D points
ax.scatter(best_points.T[:, 0], best_points.T[:, 1], best_points.T[:, 2], s=2, c="blue", label="3D Points")

# Labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Reconstruction Points")

# Save image
plt.savefig("output/given_image/3D_reconstruction.png", dpi=300)
plt.show()

