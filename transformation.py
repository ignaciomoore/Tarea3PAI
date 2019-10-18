import math

import numpy as np

def deleteArray(matrix, index):
    begining = matrix[:index]
    end = matrix[(index + 1):]
    return begining+end

def getXPrimeEcuation(x, y, xPrime, yPrime):
    return [x,y,1,0,0,0,-(x*xPrime),-(y*xPrime)]

def getYPrimeEcuation(x, y, xPrime, yPrime):
    return [0,0,0,x,y,1,-(x*yPrime),-(y*yPrime)]

def findTransformation(xSource, ySource, xDestination, yDestination):   #  4 OR MORE POINTS
    """
Finds the transformation matrix, [[a,b,c],[d,e,f],[g,h,1]]

    :param xSource: x values of 4 or more source points
    :param ySource: y values of 4 or more source points
    :param xDestination: x values of 4 or more destination points
    :param yDestination: y values of 4 or more destination points
    :return: The matrix as is shown above
    """
    ecuationMatrix = []
    destinationPointValues = []

    for point in range(4):
        ecuationMatrix.append(getXPrimeEcuation(xSource[point], ySource[point], xDestination[point], yDestination[point]))
        destinationPointValues.append(xDestination[point])
        ecuationMatrix.append(getYPrimeEcuation(xSource[point], ySource[point], xDestination[point], yDestination[point]))
        destinationPointValues.append(ySource[point])

    newEM = np.array(ecuationMatrix)
    newDPV = np.array([[destinationPointValues[0]],[destinationPointValues[1]],[destinationPointValues[2]],[destinationPointValues[3]],
                       [destinationPointValues[4]],[destinationPointValues[5]],[destinationPointValues[6]],[destinationPointValues[7]]])

    if (abs(0 - np.linalg.det(newEM)) <= 0.0001):
        return [[0,0,0],[0,0,0],[0,0,0]]
    else:
        variables = np.linalg.solve(newEM, newDPV)
        variables = np.append(variables, 1)

        transformationMatrix = np.array([variables[:3], variables[3:6], variables[6:]])

        return transformationMatrix

def findInliers(calculated, original, maxDistance):
    """
Finds the amount of inliers between two group of points

    :param calculated: The calculated group of points   a[[x,y], ...]
    :param original: The original group of points       b[[x,y], ...]
    :param maxDistance: The maximum distance between two points
    :return: the amount of pair of points whose distance is less than maxDistance
    """
    inliers = 0

    for i in range(len(calculated)):
        o = original[i]
        c = calculated[i]
        distance = calculateDistance(o,c)
        if (distance <= maxDistance):
            inliers += 1

    return inliers

def bilinear_interpolation(tl, tr, bl, br, p, Ptl, Ptr, Pbl, Pbr):
    """
Does the bilinear interpolation of the point p

    :param tl: top left (x, y) pair
    :param tr: top right (x, y) pair
    :param bl: bottom left (x, y) pair
    :param br: bottom right (x, y) pair
    :param p: point to be calculated value, (x, y) pair
    :param Ptl: top left inverse value
    :param Ptr: top right inverse value
    :param Pbl: bottom left inverse value
    :param Pbr: bottom right inverse value
    :return: point inverse value
    """
    a = p[1] - tl[1]
    b = p[0] - tl[0]

    dx = tr[0] - tl[0]
    dy = bl[1] - tl[1]

    aa = np.array([[dy - a, a]])
    bb = np.array([[dx - b], [b]])
    points = np.array([[Ptl, Ptr], [Pbl, Pbr]])
    div = dx * dy

    return float((np.matmul(aa, np.matmul(points, bb)))/float(div))

def calculateDistance(a,b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]

    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


if __name__ == '__main__':

    xp = [1, 2, 3, 4]
    yp = [1, 2, 3, 4]

    xq = [2, 3, 4, 5]
    yq = [2, 3, 4, 5]

    a = findTransformation(xp, yp, xq, yq)

    print(a)