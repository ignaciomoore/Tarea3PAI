import argparse

import cv2
import numpy as np

from ransac import ransac
from transformation import getPanoramicImageSize


parser = argparse.ArgumentParser(description='Get Panoramic Image')
parser.add_argument('--image_a', type=str, help='Image file name')
parser.add_argument('--image_b', type=str, help='Image file name')
args = parser.parse_args()

image_a_file = args.image_a
image_b_file = args.image_b


#image_a_file = 'casos/caso_1/1a.jpg'
#image_b_file = 'casos/caso_1/1b.jpg'

image_a = cv2.imread(image_a_file)
gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=6, sigma=1.6)
kp_a = sift.detect(gray_a)
kp_a, des_a = sift.compute(gray_a, kp_a)

image_b = cv2.imread(image_b_file)
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

    epsilon = 10        #   maximum distance
    iterations = 50     #   number of random transformation taken

    best_transformation = ransac(src_pts, dst_pts, epsilon, iterations)

    if np.allclose(best_transformation, np.array([[0,0,0],[0,0,0],[0,0,0]])):
        print("Didn't find transformation with distance less than " + str(epsilon) + ' and '+ str(iterations)+' iterations')
    else:
        #height = image_a.shape[0]
        #width = image_a.shape[1] + image_b.shape[1]

        newShape = getPanoramicImageSize(image_a, image_b, best_transformation)
        height = int(newShape[0])
        width = int(newShape[1])
        dah = int(newShape[2])
        daw = int(newShape[3])

        height2 = image_a.shape[0] + image_b.shape[0]
        width2 = image_a.shape[1]

        panoramic_image = np.zeros((height2, width2, 3))

        for i in range(image_a.shape[0]):
            for j in range(image_a.shape[1]):
                panoramic_image[i][j] = image_a[i][j]

        for i in range(image_b.shape[0]):
            for j in range(image_b.shape[1]):
                Q = np.array([[j], [i], [1]])
                aux = np.matmul(best_transformation, Q)
                jj = int(aux[0][0])
                ii = int(aux[1][0])

                panoramic_image[image_a.shape[0] + ii][image_a.shape[1] + jj] = image_b[i][j]

        cv2.imwrite("panoramica.png", panoramic_image)

else:
    print("Not enough good matches")
