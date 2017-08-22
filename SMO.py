import numpy as np
import readData
import matplotlib.pyplot as plt

class SMO:
    def __init__(self, dataMatIn, classLabels, C, toler, Kernel='linear', sigma=0.):
        '''
        :param dataMatIn:
        :param classLabels:
        :param C: controls the relative weighting between the twin goals of making cost small
        and of ensuring that most examples have functional margin at least 1
        :param toler:  the convergence  tolerance parameter
        :param Kernel: I will update the function later
        '''
        self.X = np.array(dataMatIn)
        self.y = np.array(classLabels).T[:, np.newaxis]
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.zeros((self.m, 1))
        self.b = 0.
        self.eCache = np.zeros((self.m, 1))
        self.K = np.zeros((self.m, self.m))
        for i in range(self.m):
            self.K[:, i] = self.kernel(self.X, self.X[i, :], Kernel, sigma)

    def kernel(self, X, Z, Kernel, sigma):
        m, n = np.shape(X)
        k = np.zeros((m, 1))
        if Kernel == 'linear':
            k = np.dot(X, Z.T)
        elif Kernel == 'rbf':
            delta = X - Z
            k = np.exp(-np.sum(delta**2, axis=1) / (sigma ** 2))
        else:
            raise NameError("we don't have this kernel\n")
        return k

    def clipAlpha(self, aj, H, L):
        '''
        :param aj:
        :param H:
        :param L:
        :return: new aj between H and L
        '''
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def calcE(self, i):
        '''
        :param i:
        :return:  a corresponding E value of ith data
        '''
        gxi = np.dot((self.alphas * self.y).T, self.K[:, i]) + self.b
        Ei = gxi - self.y[i]
        return Ei

    def selectJ(self, i, Ei):
        '''
        choose a second E value by compare others
        :param i:
        :param Ei:
        :return: best second E value
        '''
        maxK = -1
        maxDelta = 0
        Ej = 0
        self.eCache[i] = Ei
        for k in range(self.m):
            if k == i:
                continue
            Ek = self.calcE(k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDelta):
                maxK = k
                maxDelta = deltaE
                Ej = Ek
        return maxK, Ej

    def updataE(self, k):
        '''
        update new E value after change the alpha
        :param k:
        :return:
        '''
        Ek = self.calcE(k)
        self.eCache[k] = Ek

    def innerLoop(self, i):
        '''
        inner loop in data, we update alphas value each iter
        :param i:
        :return:
        '''
        Ei = self.calcE(i)
        if ((self.y[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or \
                ((self.y[i] * Ei > self.tol) and (self.alphas[i] > self.C)):
            j, Ej = self.selectJ(i, Ei)
            alphaIold = self.alphas[i].copy()
            alphaJold = self.alphas[j].copy()
            if (self.y[i] != self.y[j]):
                L = max(0, alphaJold - alphaIold)
                H = min(self.C, self.C + alphaJold - alphaIold)
            else:
                L = max(0, alphaJold + alphaIold - self.C)
                H = min(self.C, alphaJold + alphaIold)
            if (L == H):
                print("L == H")
                return 0
            eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]

            if eta <= 0:
                print("eta <= 0")
                return 0
            alphaJnewUnc = alphaJold + self.y[j] * (Ei - Ej) / eta
            alphaJnew = self.clipAlpha(alphaJnewUnc, H, L)
            self.alphas[j] = alphaJnew
            self.updataE(j)
            if (abs(alphaJnew - alphaJold) < 0.0001):
                print("j not moving enough")
                return 0
            alphaInew = alphaIold + self.y[i] * self.y[j] * (alphaJold - alphaJnew)
            self.alphas[i] = alphaInew
            self.updataE(i)
            bi = self.b - Ei - self.y[i] * self.K[i, i] * (alphaInew - alphaIold) - \
                 self.y[j] * self.K[i, j]* (alphaJnew - alphaJold)
            bj = self.b - Ej - self.y[i] * self.K[i, j] * (alphaInew - alphaIold) - \
                 self.y[j] * self.K[j, j] * (alphaJnew - alphaJold)
            if (0 < alphaInew) and (alphaInew < self.C):
                self.b = bi
            elif (0 < alphaJnew) and (alphaJnew < self.C):
                self.b = bj
            else:
                self.b = (bi + bj) / 2.0
            return 1
        else:
            return 0

    def train(self, maxIter):
        '''
        train function
        :param maxIter:
        :return: b and alphas
        '''
        iter = 0
        entireSet = True
        alphaPairsChanged = 0
        for i in range(self.m):
            Ei = self.calcE(i)
            self.eCache[i] = Ei
        while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
            alphaPairsChanged = 0
            for i in range(self.m):
                alphaPairsChanged += self.innerLoop(i)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        return self.b, self.alphas

    def calcLinearWs(self):
        w = np.sum(self.alphas * self.y * self.X, axis=0).transpose()
        # print(w.shape)
        return w

def testLinear():
    dataArr, labelArr = readData.loadDataSet('data/testSet.txt')
    smo = SMO(dataArr, labelArr, 0.6, 0.001, 'linear')
    b, alphas = smo.train(40)
    w = smo.calcLinearWs()

    X = np.array(dataArr)
    y = np.array(labelArr).T
    X_pos = X[y > 0]
    X_neg = X[y < 0]
    x = np.linspace(-7, 12, 100)

    plt.figure()
    plt.scatter(X_pos[:, 0], X_pos[:, 1], c='r')
    plt.scatter(X_neg[:, 1], X_neg[:, 1], c='g')
    print(w[0], w[1], )
    plt.plot(x[:], (w[0] * x[:] + b) / -w[1])
    plt.show()

def testRBF():
    dataArr, labelArr = readData.loadDataSet('data/testSetRBF.txt')
    smo = SMO(dataArr, labelArr, 200, 0.0001, 'rbf', 1.3)
    b, alphas = smo.train(100)
    X = np.array(dataArr)
    y = np.array(labelArr).T[:, np.newaxis]
    svInd = np.nonzero(alphas > 0)[0]
    sVs = X[svInd]
    labelSV = y[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(X)
    errorCount = 0
    # print(labelSV.shape, alphas.shape, len(b))
    for i in range(m):
        kernelEval = smo.kernel(sVs, X[i, :], 'rbf', 1.3)
        # print(np.shape(labelSV.T))
        predict = np.dot(kernelEval, labelSV * alphas[svInd]) + b
        if np.sign(predict) != np.sign(y[i]):
            errorCount += 1
    print("the training error rate is: %f" %(np.float(errorCount)/ m))
    dataArr, labelArr = readData.loadDataSet('data/testSetRBF2.txt')
    errorCount = 0
    X = np.array(dataArr)
    y = np.array(labelArr).T
    m, n = np.shape(X)
    for i in range(m):
        kernelEval = smo.kernel(sVs, X[i, :], 'rbf', 1.3)
        predict = np.dot(kernelEval.T, labelSV * alphas[svInd]) + b
        if np.sign(predict) != np.sign(y[i]):
            errorCount += 1
    print("the test error rate is: %f" % (np.float(errorCount) / m))

if __name__ == '__main__':
    # testLinear()
    testRBF()
