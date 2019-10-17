from typing import List, Any

import numpy as np

from transformation import findTransformation, findInliers, deleteArray


def ransac(P, Q, epsilon, iterations):

    pCopy = P.copy()
    qCopy = Q.copy()

    size = len(pCopy)
    subP = []
    subQ = []
    arrayIndex = 0

    bestTransformation = 0
    bestTransformationPercentage = 0

    for w in range(iterations):
        for subIndex in range(4):
            arrayIndex = np.random.randint(size - subIndex)
            subP.append(pCopy[arrayIndex])
            subQ.append(qCopy[arrayIndex])
            pCopy = pCopy[:arrayIndex] + pCopy[arrayIndex+1 :]# deleteArray(pCopy, arrayIndex)
            qCopy = qCopy[:arrayIndex] + qCopy[arrayIndex+1 :]# deleteArray(qCopy, arrayIndex)

        xSubP = [subP[0][0],subP[1][0],subP[2][0],subP[3][0]]
        ySubP = [subP[0][1],subP[1][1],subP[2][1],subP[3][1]]
        xSubQ = [subQ[0][0],subQ[1][0],subQ[2][0],subQ[3][0]]
        ySubQ = [subQ[0][1],subQ[1][1],subQ[2][1],subQ[3][1]]

        inverseTransfromation = findTransformation(xSubQ, ySubQ, xSubP, ySubP)

        if (inverseTransfromation == 0):
            print("Inverse Matrix not available")
        else:
            calculatedSourceP = []

            for i in range(len(Q)):
                calculatedSourceP.append(np.matmul(inverseTransfromation, np.array([[Q[i][0]],[Q[i][1]],[1]])))

            inliers = findInliers(calculatedSourceP, P, epsilon)

        percentage = (inliers*100)/len(P)

        if percentage > bestTransformationPercentage:
            bestTransformationPercentage = percentage
            bestTransformation = inverseTransfromation

    return bestTransformation

if __name__ == '__main__':

    a = [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]]
    b = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8]]

    ransac(a,b,2,4)
