import cv2
import matplotlib.pyplot as plt
from Config import GIVEN_DATA_IMG1, GIVEN_DATA_IMG2

# Read the image
image1 = cv2.imread(GIVEN_DATA_IMG1)
image2 = cv2.imread(GIVEN_DATA_IMG2)

# Convert to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Create the SIFT Object
sift = cv2.SIFT_create(10)

# Detect the keypoints and calculate the descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Draw keypoints in the images
image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the images with keypoint (SIFT detector)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Image 1 with Keypoints')
plt.imshow(cv2.cvtColor(image1_with_keypoints, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Image 2 with Keypoints')
plt.imshow(cv2.cvtColor(image2_with_keypoints, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
