import numpy as np


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


def inverse_transformation(source_points, destination_points):    #    implement for all points in dest_pts
    return 0


if __name__ == '__main__':

    tl = (0, 0)
    tr = (2, 0)
    bl = (0, 2)
    br = (2, 2)

    p = (1, 1)

    Ptl = 1
    Ptr = 1
    Pbl = 3
    Pbr = 3

    print(bilinear_interpolation(tl, tr, bl, br, p, Ptl, Ptr, Pbl, Pbr))