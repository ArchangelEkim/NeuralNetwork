# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:52:19 2019

@author: micke
"""

import numpy as np

class NeuralNetwork:

    def __init__(self, layer_sizes):
        weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s)/s[1]**.5 for s in weight_shapes]
        self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]]

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for w,b in zip(self.weights, self.biases):
            a = self.sigmoid(np.matmul(w,a) + b)
        return a

    def print_accuracy(self, images, labels):
        predictions = self.feedforward(images)
        num_correct = sum([np.argmax(a) == np.argmax(b) for a,b in zip(predictions, labels)])
        print('{0}/{1} accuracy: {2}%'.format(num_correct, len(images), round(num_correct/len(images)*100, 4)))

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))    