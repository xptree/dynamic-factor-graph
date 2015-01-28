#!/usr/bin/env python

# encoding: utf-8
# File Name: dfg.py
# Author: Jiezhong Qiu
# Create Time: 2015/01/26 00:25
# TODO:

from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator
import logging
import time
import datetime
import os
import cPickle as pickle
import factor
import unittest


logger = logging.getLogger(__name__)
theano.config.exception_verbosity='high'
#mode = theano.Mode(linker='cvm')
#mode = 'DebugMode'
mode = 'FAST_COMPILE'
#mode = 'ProfileMode'

class DFG(object):
    """     Dynamic factor graph class

    Support output types:
    real: linear output units, use mean-squared error
    binary: binary output units, use cross-entropy error
    softmax: single softmax out, use cross-entropy error
    """
    def __init__(self, input, n_hidden, n_obsv, n_step, order=1,
                output_type='real', factor_type='FIR'):
        self.input = input
        self.n_hidden = n_hidden
        self.n_obsv = n_obsv
        self.n_step = n_step
        self.order = order
        self.output_type = output_type
        self.factor_type = factor_type
        if self.factor_type == 'FIR':
            self.factor = factor.FIR(n_hidden=self.n_hidden,
                                        n_obsv=self.n_obsv, n_step=self.n_step, order=self.order, hmm=True)
        else:
            raise NotImplementedError

        self.params_Estep = self.factor.params_Estep
        self.params_Mstep = self.factor.params_Mstep
        self.L1 = self.factor.L1
        self.L2_sqr = self.factor.L2_sqr

        self.y_pred = self.factor.y_pred
        self.z_pred = self.factor.z_pred
        self.z = self.factor.z

        self.updates = OrderedDict()
        for param in self.params_Estep:
            init = np.zeros(param.get_value(borrow=True).shape,
                    dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)
        for param in self.params_Mstep:
            init = np.zeros(param.get_value(borrow=True).shape,
                    dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)


        if self.output_type == 'real':
            self.loss = lambda y : self.mse(y)
        else:
            raise NotImplementedError

    def mse(self, y):
        # error between output and target
        # error between predicted latent and target latent
        return T.mean((self.y_pred - y) ** 2) + T.mean((self.z_pred - self.z[self.order:]) ** 2)


class MetaDFG(BaseEstimator):
    def __init__(self, n_hidden, n_obsv, n_step, order, learning_rate=0.01,
                n_epochs=1000, batch_size=100, L1_reg=0.00, L2_reg=0.00,
                learning_rate_decay=1,
                factor_type='FIR', activation='tanh', output_type='real', final_momentum=0.9,
                initial_momentum=0.5, momentum_switchover=5,
                snapshot_every=None, snapshot_path='/tmp'):
        self.n_hidden = int(n_hidden)
        self.n_obsv = int(n_obsv)
        self.n_step = int(n_step)
        self.order = int(order)
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.factor_type = factor_type
        self.activation = activation
        self.output_type = output_type
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)
        if snapshot_every is not None:
            self.snapshot_every = int(snapshot_every)
        else:
            self.snapshot_every = None
        self.snapshot_path = snapshot_path
        self.ready()

    def ready(self):
        # observation (where first dimension is time)
        if self.output_type == 'real':
            self.y = T.matrix(name='y', dtype=theano.config.floatX)
        else:
            raise NotImplementedError

        # learning rate
        self.lr = T.scalar()

        if self.activation == 'tanh':
            activation = T.tanh
        elif self.activation == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activation == 'relu':
            activation = lambda x: x * (x > 0)
        else:
            raise NotImplementedError

        self.dfg = DFG(input=self.y, n_hidden=self.n_hidden,
                        n_obsv=self.n_obsv, n_step=self.n_step,
                        order=self.order, output_type=self.output_type,
                        factor_type=self.factor_type)

        if self.output_type == 'real':
            self.predict = theano.function(inputs=[],
                                           outputs=self.dfg.y_pred,
                                           mode=mode)
        else:
            raise NotImplementedError
    def shared_dataset(self, data):
        """ Load the dataset into shared variables """

        shared_data = theano.shared(np.asarray(data,
                                            dtype=theano.config.floatX))
        return shared_data

    def fit(self, Y_train, Y_test=None,
            validation_frequency=100):
        """Fit model
            Y_train : ndarray (n_seq, n_t, n_out)
        """


        self.interactive = False
        train_set_y = self.shared_dataset(Y_train)
        n_train = train_set_y.get_value(borrow=True).shape[0]

        logger.info('...building the model')

        index = T.lscalar('index') #index to a case
        # learning rate (may change)
        l_r = T.scalar('l_r', dtype=theano.config.floatX)
        mom = T.scalar('mom', dtype=theano.config.floatX)

        cost = self.dfg.loss(self.y) \
                + self.L1_reg * self.dfg.L1 \
                + self.L2_reg * self.dfg.L2_sqr

        # compute the gradient of cost with respect to theta = (W, W_in, W_out)
        # gradients on the weights using BPTT
        # E step
        gparams_Estep = []
        for param in self.dfg.params_Estep:
            gparam = T.grad(cost, param)
            gparams_Estep.append(gparam)

        updates_Estep = OrderedDict()
        for param, gparam in zip(self.dfg.params_Estep, gparams_Estep):
            weight_update = self.dfg.updates[param]
            upd = mom * weight_update - l_r * gparam
            updates_Estep[weight_update] = upd
            updates_Estep[param] = param + upd

        # M step
        gparams_Mstep = []
        for param in self.dfg.params_Mstep:
            gparam = T.grad(cost, param)
            gparams_Mstep.append(gparam)

        updates_Mstep = OrderedDict()
        for param, gparam in zip(self.dfg.params_Mstep, gparams_Mstep):
            weight_update = self.dfg.updates[param]
            upd = mom * weight_update - l_r * gparam
            updates_Mstep[weight_update] = upd
            updates_Mstep[param] = param + upd

        # compiling a Theano function `train_model_Estep` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates_Estep`
        train_model_Estep = theano.function(inputs=[index, l_r, mom],
                                        outputs=cost,
                                        updates=updates_Estep,
                                        givens=OrderedDict([(self.y, train_set_y[index])]),
                                        mode=mode)
        # updates the parameter of the model based on
        # the rules defined in `updates_Mstep`
        train_model_Mstep = theano.function(inputs=[index, l_r, mom],
                                        outputs=cost,
                                        updates=updates_Mstep,
                                        givens=OrderedDict([(self.y, train_set_y[index])]),
                                        mode=mode)

        ###############
        # TRAIN MODEL #
        ###############
        logger.info('... training')
        epoch = 0

        while (epoch < self.n_epochs):
            epoch = epoch + 1
            average_cost = 0.
            for idx in xrange(n_train):
                effective_momentum = self.final_momentum \
                               if epoch > self.momentum_switchover \
                               else self.initial_momentum
                example_cost = train_model_Estep(idx, self.learning_rate,
                                           effective_momentum)
                average_cost += example_cost
            logger.info('epoch %d E_step cost=%f' % (epoch, average_cost / n_train))
            average_cost = 0.
            for idx in xrange(n_train):
                effective_momentum = self.final_momentum \
                               if epoch > self.momentum_switchover \
                               else self.initial_momentum
                example_cost = train_model_Mstep(idx, self.learning_rate,
                                            effective_momentum)
                average_cost += example_cost
            logger.info('epoch %d M_step cost=%f' % (epoch, average_cost / n_train))
            self.learning_rate *= self.learning_rate_decay


class sinTestCase(unittest.TestCase):
    def runTest(self):
        n = 100
        x = np.linspace(100, 200, n)
        sita = [.2, .331, .42, .51, .74]
        y = np.zeros(n)
        for item in sita:
            y += np.sin(item * x)
        # n_seq x n_t x n_in
        y = y.reshape(1, n, 1)
        dfg = MetaDFG(n_hidden=5, n_obsv=1, n_step=n, order=11)
        dfg.fit(y)
        assert True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

