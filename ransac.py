from typing import List, Any

import numpy as np
import random

from transformation import findTransformation, findInliers, deleteArray


def ransac(P, Q, epsilon, iterations):

    pCopy = P.copy()
    qCopy = Q.copy()

    size = len(pCopy)
    indexShuffle = []
    for i in range(len(P)):
        indexShuffle.append(i)
    random.shuffle(indexShuffle)

    subP = []
    subQ = []

    bestTransformation = [[0,0,0],[0,0,0],[0,0,0]]
    bestTransformationPercentage = 0

    for w in range(iterations):
        for subIndex in range(4):
            subP.append(pCopy[indexShuffle[subIndex]])
            subQ.append(qCopy[indexShuffle[subIndex]])

        # x = j, y = i
        xSubP = [subP[0][1], subP[1][1], subP[2][1], subP[3][1]] # j
        ySubP = [subP[0][0], subP[1][0], subP[2][0], subP[3][0]] # i
        xSubQ = [subQ[0][1], subQ[1][1], subQ[2][1], subQ[3][1]]
        ySubQ = [subQ[0][0], subQ[1][0], subQ[2][0], subQ[3][0]]

        inverseTransfromation = findTransformation(xSubQ, ySubQ, xSubP, ySubP) # use like [[j],[i],[z]]

        if (inverseTransfromation.all() == 0):
            print("Inverse Matrix not available")
        else:
            calculatedSourceP = []

            for i in range(len(Q)):
                aux = np.matmul(inverseTransfromation, np.array([[Q[i][0]],[Q[i][1]],[1]]))
                x = aux[0][0]
                y = aux[1][0]
                aux = [x,y]
                calculatedSourceP.append(aux[:2])

            inliers = findInliers(np.array(calculatedSourceP), P, epsilon)

        percentage = (inliers*100)/len(P)

        if percentage > bestTransformationPercentage:
            bestTransformationPercentage = percentage
            bestTransformation = inverseTransfromation

    return bestTransformation

if __name__ == '__main__':

    a = [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]]
    b = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8]]

    ransac(a,b,2,4)
