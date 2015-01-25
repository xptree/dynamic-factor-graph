#!/usr/bin/env python
# encoding: utf-8
# File Name: dfg.py
# Author: Jiezhong Qiu
# Create Time: 2015/01/26 00:25
# TODO:

import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator
import logging
import time
import datetime
import os
import cPickle as pickle


logger = logging.getLogger(__name__)

mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'

class DFG(object):
    """     Dynamic factor graph class

    Support output types:
    real: linear output units, use mean-squared error
    binary: binary output units, use cross-entropy error
    softmax: single softmax out, use cross-entropy error
    """
    def __init__(self, input, n_in, n_hidden, n_out, order=1,
                output_type='real', factor = None):
        self.input = input
        self.output_type = output_type

        [self.z, self.y_pred], _ = theano.scan(factor.onestep,
                                    sequences=[ dict(input=self.input, taps=range(-order, 1)) ],
                                    outputs_info = [dict(initial = self.z, taps=range(-order, 0)), None])
        self.params = factor.params
        self.L1 = factor.L1
        self.L2_sqr = factor.L2

        if self.output_type == 'real':
            self.loss = lambda y : self.mse(y)
        elif self.output_type == 'binary':
            # push through sigmoid
            self.p_y_given_x = T.nnet.sigmoid(self.y_pred)  # apply sigmoid
            self.y_out = T.round(self.p_y_given_x)  # round to {0,1}
            self.loss = lambda y: self.nll_binary(y)
        elif self.output_type == 'softmax':
            # push through softmax, computing vector of class-membership
            # probabilities in symbolic form
            self.p_y_given_x = self.softmax(self.y_pred)
            # compute prediction as class whose probability is maximal
            self.y_out = T.argmax(self.p_y_given_x, axis=-1)
            self.loss = lambda y: self.nll_multiclass(y)
        else:
            raise NotImplementedError

    def mse(self, y):
        # error between output and target
        return T.mean((self.y_pred - y) ** 2)

    def nll_binary(self, y):
        # negative log likelihood based on binary cross entropy error
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

    def nll_multiclass(self, y):
        # negative log likelihood based on multiclass cross entropy error
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of time steps (call it T) in the sequence
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

if __name__ == "__main__":
    pass


