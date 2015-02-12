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
    def __init__(self, n_in, n_hidden, n_obsv, n_step, order, n_seq, start, n_iter):
        self.n_in = n_in
        self.n_hiddent = n_hidden
        self.n_obsv = n_obsv
        self.n_step = n_step
        self.order = order
        self.n_seq = n_seq
        self.start = start # symbolic
        self.n_iter = n_iter # symbolic
        # initialize np.random
        #np.random.seed(20)
        # we consider g as linear observation models
        # Y(t) = g(W_o, Z(t))
        W_o_n_in = n_hidden
        W_o_n_out = n_obsv
        W_o_bound = 4 * np.sqrt(6. / (W_o_n_in + W_o_n_out))
        W_o_init = np.asarray(np.random.uniform(size=(n_hidden, n_obsv),
                                    low=-W_o_bound, high=W_o_bound),
                                    dtype=theano.config.floatX)
        self.W_o = theano.shared(value=W_o_init, name='W_o')
        b_o_init = np.zeros((n_obsv,), dtype=theano.config.floatX)
        self.b_o = theano.shared(value=b_o_init, name='b_o')
        #z0_init = np.zeros(size=(order, n_hidden), dtype=theano.config.floatX)
        #self.z0 = theano.shared(value=z0_init, name='z0')
        z_init = np.asarray(np.random.uniform(size=(n_step + order, n_seq, n_hidden),
                                low=0., high=1.0),
                                dtype=theano.config.floatX)
        self.z = theano.shared(value=z_init, name='z')
        self.params_Estep = [self.z]
        self.params_Mstep = [self.W_o, self.b_o]
        self.L1 = abs(self.W_o).sum()
                #+ abs(self.b_o).sum()
                #+ abs(self.z).sum()
        self.L2_sqr = (self.W_o ** 2).sum()
                #+ (self.b_o ** 2).sum()
                #+ (self.z ** 2).sum()

class FIR(Factor):
    def __init__(self, n_in, n_hidden, n_obsv, n_step, order, n_seq, start, n_iter, batch_start, batch_stop):
        logger.info('A FIR factor built ...')
        Factor.__init__(self, 0, n_hidden, n_obsv, n_step, order, n_seq, start, n_iter)
        W_bound = order
        W_init = np.asarray(np.random.uniform(size=(order*n_hidden, n_hidden),
                                low=-1.0 / W_bound, high=1.0 / W_bound),
                                dtype=theano.config.floatX)
        self.W = theano.shared(value=W_init, name="W")
        self.params_Mstep.append(self.W)
        self.L1 += abs(self.W).sum()
        self.L2_sqr += (self.W ** 2).sum()

        def step(*args):
            """
                z_tmp, ..., z_tm1 \in R^{batch_size, n_hidden}
            """
            z_concatenate = T.concatenate(args, axis=1) # (batch_size, n_hidden x order)
            z_t = T.dot(z_concatenate, self.W)
            y_t = T.dot(z_t, self.W_o) + self.b_o
            return z_t, y_t

        # z_pred, y_pred for E_step
        [self.z_pred, self.y_pred], _ = theano.scan(step,
                                    sequences=[ dict(input=self.z, taps=range(-order, 0)) ])

        self.z_subtensor = self.z[self.start:self.start+order,batch_start:batch_stop]
        [self.z_next, self.y_next], _ = theano.scan(step,
                                    n_steps=self.n_iter,
                                    outputs_info = [ dict(initial=self.z_subtensor, taps=range(-order, 0)) , None ])
class MLP(Factor):
    def __init__(self, n_in, x, y_tm1, n_hidden, n_obsv, n_step, order, n_seq, start, n_iter, batch_start, batch_stop, no_past_obsv=True):
        logger.info('A MLP factor built ...')
        Factor.__init__(self, n_in, n_hidden, n_obsv, n_step, order, n_seq, start, n_iter)
        # real value passed through givens=[..]
        self.x = x
        self.y_tm1 = y_tm1
        if no_past_obsv:
            W_n_in = order * n_hidden + n_in
        else:
            W_n_in = order * n_hidden + n_in + n_obsv
        W_n_out = n_hidden
        W_bound = 4 * np.sqrt(6. / (W_n_in + W_n_out))
        W_init = np.asarray(np.random.uniform(size=(W_n_in, W_n_out),
                                low=-W_bound, high=W_bound),
                                dtype=theano.config.floatX)
        self.W = theano.shared(value=W_init, name="W")
        b_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_init, name='b')
        self.params_Mstep.append(self.W)
        self.params_Mstep.append(self.b)
        self.L1 += abs(self.W).sum()
        self.L2_sqr += (self.W ** 2).sum()

        def step(*args):
            """
                args include
                x_t \in R^{batch_size, n_in}
                z_tmp, ..., z_tm1 \in R^{batch_size, n_hidden}
                y_tm1 \in R^{batch_size, n_obsv}
            """
            z_concatenate = T.concatenate(args, axis=1) # (batch_size, n_hidden x order + n_in + n_obsv)
            z_t = T.nnet.sigmoid(T.dot(z_concatenate, self.W) + self.b)
            y_t = T.nnet.sigmoid(T.dot(z_t, self.W_o) + self.b_o)
            return z_t, y_t

        # Compute z_pred, y_pred for E_step
        # Here x should be T x n_seq x n_in
        # and y_tm1 should be T x n_seq x n_obsv
        sequences=[ dict(input=self.x, taps=[0]),
                    dict(input=self.z, taps=range(-order, 0)),
                    dict(input=self.y_tm1, taps=[0]) ]
        if no_past_obsv:
            sequences = sequences[:-1]
        [self.z_pred, self.y_pred], _ = theano.scan(step,
                                                    sequences=sequences)
        self.z_subtensor = self.z[self.start:self.start+order,batch_start:batch_stop]

        # Compute z_next, y_next for either M step or performance evaluation
        # Here x should be n_iter x effective_batch_size x n_in
        # and y_tm1 should be 1 x effective_batch_size x n_obsv

        outputs_info = [ dict(initial=self.z_subtensor, taps=range(-order, 0)),
                            dict(initial=self.y_tm1[-1]) ]
        if no_past_obsv:
            outputs_info[-1] = None
        [self.z_next, self.y_next], _ = theano.scan(step,
                                    sequences=[ dict(input=self.x, taps=[0]) ],
                                    n_steps=self.n_iter,
                                    outputs_info=outputs_info)
if __name__ == "__main__":
    pass


