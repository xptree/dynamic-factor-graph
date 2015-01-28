#!/usr/bin/env python

# encoding: utf-8
# File Name: factor.py
# Author: Jiezhong Qiu
# Create Time: 2015/01/26 03:05
# TODO:

import numpy as np
import theano
import theano.tensor as T
import logging

logger = logging.getLogger(__name__)

class Factor(object):
    def __init__(self, n_hidden, n_obsv, n_step, order = 1, hmm=True):
        self.n_hiddent = n_hidden
        self.n_obsv = n_obsv
        self.n_step = n_step
        self.order = order
        self.hmm = hmm
        # we consider g as linear observation models
        # Y(t) = g(W_o, Z(t))
        W_o_init = np.asarray(np.random.uniform(size=(n_hidden, n_obsv),
                                    low=-.01, high=.01),
                                    dtype=theano.config.floatX)
        self.W_o = theano.shared(value=W_o_init, name='W_o')
        b_o_init = np.zeros((n_obsv,), dtype=theano.config.floatX)
        self.b_o = theano.shared(value=b_o_init, name='b_o')
        #z0_init = np.zeros(size=(order, n_hidden), dtype=theano.config.floatX)
        #self.z0 = theano.shared(value=z0_init, name='z0')
        z_bound = np.sqrt((n_step + order) * n_hidden)
        z_init = np.asarray(np.random.uniform(size=(n_step + order, n_hidden),
                                low=-1.0 / z_bound, high=1. / z_bound),
                                dtype=theano.config.floatX)
        self.z = theano.shared(value=z_init, name='z')
        self.params_Estep = [self.z]
        self.params_Mstep = [self.W_o, self.b_o]
        self.L1 = abs(self.W_o.sum())
        self.L2_sqr = (self.W_o ** 2).sum()

class FIR(Factor):
    def __init__(self, n_hidden, n_obsv, n_step, order = 1, hmm=True):
        Factor.__init__(self, n_hidden, n_obsv, n_step, order, hmm)
        if hmm == False:
            raise NotImplementedError
        W_bound = np.sqrt(order * n_hidden)
        W_init = np.asarray(np.random.uniform(size=(order, n_hidden),
                                low=-1.0 / W_bound, high=1.0 / W_bound),
                                dtype=theano.config.floatX)
        self.W = theano.shared(value=W_init, name="W")
        self.params_Mstep.append(self.W)
        self.L1 += abs(self.W.sum())
        self.L2_sqr += (self.W ** 2).sum()

        def step(*args):
            """
                z_tmp, ..., z_tm1 \in R^{1,n_hidden}
            """
            z_stack = T.stacklists(args)
            z_merge = z_stack * self.W
            z_t = T.sum(z_merge, axis=0)
            y_t = T.dot(z_t, self.W_o) + self.b_o
            return z_t, y_t

        [self.z_pred, self.y_pred], _ = theano.scan(step,
                                    sequences=[ dict(input=self.z, taps=range(-order, 0)) ])

#        def onestep(z_tm1, z_tm0):
#            z_t = T.sum(z_tm1 * self.W, axis=0)
#            y_t = T.dot(z_t, self.W_o) + self.b_o
#            return z_t, y_t
#        # implement a hmm version, thus order == 1
#        [self.z_pred, self.y_pred], _ = theano.scan(onestep,
#                                    sequences=[ dict(input=self.z, taps=[-1, -0]) ])
if __name__ == "__main__":
    pass


