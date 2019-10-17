
import cv2
import numpy as np

from ransac import ransac

image_a = cv2.imread('casos/caso_1/1a.jpg')
gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=6, sigma=1.6)
kp_a = sift.detect(gray_a)
kp_a, des_a = sift.compute(gray_a, kp_a)

image_b = cv2.imread('casos/caso_1/1b.jpg')
gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
kp_b = sift.detect(gray_b)
kp_b, des_b = sift.compute(gray_b, kp_b)

matcher = cv2.BFMatcher()
matches = matcher.knnMatch(des_a, des_b, k=2)

good = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good.append(m)

if len(good) > 4:
    src_pts = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 2)

    best_transformation = ransac(src_pts, dst_pts, 4, 10)

    print(best_transformation)

else:
    print("Not enough good matches")
