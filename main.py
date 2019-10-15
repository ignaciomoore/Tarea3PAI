#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

img_a = cv2.imread('casos/caso_1/1a.jpg')
gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=6, sigma=1.6)
print(type(sift))
kp = sift.detect(gray)
kp, des = sift.compute(gray, kp)

img_b = cv2.imread('casos/caso_1/1b.jpg')
gray_2 = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
kp_2 = sift.detect(gray_2)
kp_2, des_2 = sift.compute(gray_2, kp_2)

matcher = cv2.BFMatcher()
matches = matcher.knnMatch(des, des_2, k=2)
# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good.append(m)

##Calculando homografía para afinar correspondencias
if len(good) > 4:
    src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst_pts = np.float32([kp_2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)    #    Implement bilinear interpolation
    matchesMask = mask.ravel().tolist()
else:
    matchesMask = None

# Aquí, matchesMask contiene las correspondencias
final_image = cv2.drawMatches(img_a, kp, img_b, kp_2, good, None, flags=2, matchesMask=matchesMask)

cv2.imshow("matches", final_image)
cv2.waitKey()