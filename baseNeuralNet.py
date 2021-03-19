import numpy as np
import random

class BaseNeuralNet:
    def __init__(self, inputDim):
        self.layers = [inputDim]
        self.params = {}

    def addLayer(self, numNodes):
        self.layers.append(numNodes)
        L = len(self.layers)
        layers = self.layers
        self.params["W"+str(L-1)] = np.random.randn(layers[L-1], layers[L-2])
        self.params["b"+str(L-1)] = np.random.randn(layers[L-1], 1)

    def sigmoid(self, a):
        return 1/(1 + np.exp(-1 * a))

    def mse(self, y, yt):
        cost = 0.5*(yt-y)**2
        cost = cost/(y.shape[0]*y.shape[1])
        return cost

    def feed(self, X):
        A = [0]
        Z = [0]
        #y=xw+b
        L = len(self.layers)
        for i range(1, L):
            Wi = self.params["W"+str(i)]
            bi = self.params["b"+str(i)]
            Zi = np.dot(X, Wi.T) + bi
            Z.append(Zi)
            Ai = self.sigmoid(Zi)
            A.append(Ai)
        return A, Z

    def backward_prop(self, X, Y, cache):
        grads = {}
        L = len(self.params)//2
        m = Y.shape[1]
        AL = cache["A"+str(L)]
        dZ = np.multiply(AL - Y, np.multiply(AL, 1 - AL))
        grads["dW" + str(L)] = (1/m) * np.dot(dZ, cache["A" + str(L - 1)].T)
        grads["db" + str(L)]=(1/m) * np.sum(dZ, axis = 1, keepdims = True)
        cache["A0"] = X
        for i in range(L-1, 0, -1):
            y = cache["A" + str(i)]
            dZL_1 = np.dot(self.params["W" + str(i + 1)].T, dZ) * (y - np.power(y, 2))
            grads["dW" + str(i)] = (1/m) * np.dot(dZL_1, cache["A" + str(i - 1)].T)
            grads["db" + str(i)] = (1/m) * np.sum(dZL_1, axis = 1, keepdims = True)
            dZ = dZL_1

        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.params) // 2
        for i in range(1, L + 1):
            self.params["W" + str(i)] = self.params["W"+str(i)] - (learning_rate * grads["dW" + str(i)])
            self.params["b" + str(i)] = self.params["b" + str(i)] - (learning_rate * grads["db" + str(i)])

    def predict(self, X):
        A = X
        L = len(self.params) // 2
        for i in range(1, L + 1):
            A = np.dot(self.params["W" + str(i)], A) + self.params["b" + str(i)]
            A = self.sigmoid(A)
        return A

    def train(self, X, Y, n, lr):
        for i in range(n): 
            AL, cache = self.feed(X)
            cost = self.mse(AL,Y)
            if i%10 == 0:
                print("Cost after " + str(i) + ":" + str(cost))
            grads = self.backward_prop(X, Y, cache)
            self.update_parameters(grads, lr)
            
    
    def saveWeights(self, name):
        f = open(name, "wb")
        pickle.dump(self.params, f)
        f.close()

    def getWeights(self, name):
        f = open(name, "rb")
        self.params = pickle.load(f)
        f.close()


