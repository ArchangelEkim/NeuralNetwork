# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:41:54 2019

@author: micke
"""

import numpy as np

layer_sizes = (2,1)

x = [-1,  0, 10]
y = [ 1, -1,  1]
inputs = [np.array(((x),(y))) for x, y in zip(x,y)]
results = [1, -1, 1]
outputs = [np.array(((x))) for z in results]
data = [(a,b) for a, b in zip(inputs,outputs)]



class bNeuralNetwork:
    
    def __init__(self, layer_sizes):

        weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.zeros(s) for s in weight_shapes]
        self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]]

    def perceptlearn(self, data):
        for x, y in 3*data:
#            print(x)
#            print(y)
            if self.indicator(x, y):
                self.weights += x*y
                self.biases += y
    
    def indicator(self, x, y):
        connection = np.matmul(self.weights, x) + self.biases
        if connection*y <= 0:
            return True
        else:
            return False
        