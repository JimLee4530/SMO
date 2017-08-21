import numpy as np
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([np.float(lineArr[0]), np.float(lineArr[1])])
        labelMat.append(np.float(lineArr[2]))
    return dataMat, labelMat