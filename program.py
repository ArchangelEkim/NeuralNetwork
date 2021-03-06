# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:48:30 2019

@author: micke
"""

import neuralnetwork_example as nn
import numpy as np
import time

start_time = time.time()

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

#plt.imshow(training_images[0].reshape(28,28), cmap = 'gray')
#print(training_images[0])
training_data = [(training_images[i],training_labels[i]) for i in range(len(training_images))]
test_data = [(test_images[i],test_labels[i]) for i in range(len(test_images))]

layer_sizes = (784,64,10)

net = nn.NeuralNetwork(layer_sizes)
net.SGD(training_data, 30, 10, 3.0, test_data)

end_time = time.time()
print('Finished in {}s'.format(round(end_time - start_time, 2)))