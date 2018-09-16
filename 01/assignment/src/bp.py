#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def backprop(x, y, biases, weights, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_w): tuple containing the gradient for all the biases
                and weights. nabla_b and nabla_w should be the same shape as 
                input biases and weights
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    ###
    #
    a =[np.zeros(b.shape) for b in biases]
    h_s = [np.zeros(b.shape) for b in biases]
    b = [np.zeros(c.shape) for c in biases]
    h0=[x]
    activations=h0+h_s

    #print(a)
    #print(weights)
    #print(weights[0])
    for k in range(2):
       a[k]= biases[k] + np.dot(weights[k],activations[k])
       activations[k+1] = sigmoid(a[k])



    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).delta(activations[-1], y)

    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###
    for k in range(num_layers, -1, -1):



    return (nabla_b, nabla_w)

