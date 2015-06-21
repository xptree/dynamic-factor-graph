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

class MLP(Factor):
    def __init__(self, n_in, x, y_pad, n_hidden, n_obsv, n_step, order, n_seq, start, n_iter, batch_start, batch_stop, order_obsv=0, hidden_layer_config=None):
        logger.info('A MLP factor with %d order obsv and %d order latent states built ...' % (order_obsv, order))
        Factor.__init__(self, n_in, n_hidden, n_obsv, n_step, order, n_seq, start, n_iter)
        # real value passed through givens=[..]
        self.x = x
        self.y_pad = y_pad
        W_n_in = order * n_hidden + n_in + order_obsv * n_obsv
        W_n_out = n_hidden
        if hidden_layer_config is None:
            W_n_hidden = (W_n_in + W_n_out) * 2 / 3
            layer_size = [W_n_in, W_n_hidden, W_n_out]
        else:
            layer_size = [W_n_in] + hidden_layer_config + [W_n_out]
        logger.info('MLP layer sizes: %s' % ' '.join([str(item) for item in layer_size]))
        self.Ws, self.bs = [], []
        for i in xrange(len(layer_size) - 1):
            W_bound = 4 * np.sqrt(6. / (layer_size[i] + layer_size[i+1]))
            W_init = np.asarray(np.random.uniform(size=(layer_size[i], layer_size[i+1]),
                                low=-W_bound, high=W_bound),
                                dtype=theano.config.floatX)
            W = theano.shared(value=W_init, name='W_%d' % i)
            b_init = np.zeros((layer_size[i+1],), dtype=theano.config.floatX)
            b = theano.shared(value=b_init, name='b_%d' % i)
            self.params_Mstep.append(W)
            self.params_Mstep.append(b)
            self.L1 += abs(W).sum()
            self.L2_sqr += (W ** 2).sum()
            self.Ws.append(W)
            self.bs.append(b)

        #rng = np.random.RandomState()
        #self.srng = T.shared_randomstreams.RandomStreams(
        #                    rng.randint(999999))

        def step(*args):
            """
                args include
                x_t \in R^{batch_size, n_in}
                z_tmp, ..., z_tm1 \in R^{batch_size, n_hidden}
                y_pad \in R^{batch_size, n_obsv}
            """
        #    args = list(args)
        #    for i in xrange(order_obsv):
        #        args[-i-1] = args[-i-1][:,:-1]
            z_t = T.concatenate(args, axis=1) # (batch_size, n_hidden x order + n_in + n_obsv)
            for i in xrange(len(layer_size) - 1):
                this_W = self.Ws[i]
                this_b = self.bs[i]
                z_t = T.nnet.sigmoid(T.dot(z_t, this_W) + this_b)
            y_t = T.nnet.sigmoid(T.dot(z_t, self.W_o) + self.b_o)
            return z_t, y_t

        # Compute z_pred_Estep, y_pred_Estep for E_step
        # Here x should be T x n_seq x n_in
        # and y_pad should be T x n_seq x n_obsv
        sequences=[ dict(input=self.x, taps=[0]),
                    dict(input=self.z, taps=range(-order, 0)),
                    dict(input=self.y_pad, taps=range(-order_obsv, 0)) ]
        if order_obsv == 0:
            sequences = sequences[:-1]
        [self.z_pred_Estep, self.y_pred_Estep], _ = theano.scan(step,
                                                                sequences=sequences)

        self.z_subtensor = self.z[self.start:self.start+order+n_iter,batch_start:batch_stop]
        self.y_subtensor = self.y_pad[self.start:self.start+order_obsv+n_iter, batch_start:batch_stop]
        sequences=[ dict(input=self.x, taps=[0]),
                    dict(input=self.z_subtensor, taps=range(-order, 0)),
                    dict(input=self.y_subtensor, taps=range(-order_obsv, 0)) ]
        if order_obsv == 0:
            sequences = sequences[:-1]
        [self.z_pred_Mstep, self.y_pred_Mstep], _ = theano.scan(step,
                                                                sequences=sequences)

        # Compute z_next, y_next for either M step or performance evaluation
        # Here x should be n_iter x effective_batch_size x n_in
        # and y_pad should be 1 x effective_batch_size x n_obsv

        self.z_subtensor = self.z[self.start:self.start+order,batch_start:batch_stop]
        self.y_subtensor = self.y_pad[self.start:self.start+order_obsv,batch_start:batch_stop]
        outputs_info = [ dict(initial=self.z_subtensor if order > 1 else self.z_subtensor[0], taps=range(-order, 0)),
                            dict(initial=self.y_subtensor if order_obsv > 1 else self.y_subtensor[0], taps=range(-order_obsv, 0)) ]
        if order_obsv == 0:
            outputs_info[-1] = None
        [self.z_next, self.y_next], _ = theano.scan(step,
                                    sequences=[ dict(input=self.x, taps=[0]) ],
                                    n_steps=self.n_iter,
                                    outputs_info=outputs_info)
if __name__ == "__main__":
    pass


