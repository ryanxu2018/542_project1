#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: activation.py

import numpy as np
from scipy.misc import derivative as dev

def sigmoid(z):
    """The sigmoid function."""
    return (1/(1+np.exp(-z)))

def sigmoid_prime(z):
    """return derivative of sigmoid function"""
    out = sigmoid(z)
    return out*(1-out)

