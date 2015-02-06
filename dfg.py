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
import matplotlib.pylab as plt


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
    def __init__(self, input, n_hidden, n_obsv, n_step, order, start, n_iter,
                factor_type='FIR'):
        self.input = input
        self.n_hidden = n_hidden
        self.n_obsv = n_obsv
        self.n_step = n_step
        self.order = order
        self.factor_type = factor_type
        # For mini-batch
        self.start = start
        self.n_iter = n_iter
        if self.factor_type == 'FIR':
            self.factor = factor.FIR(n_hidden=self.n_hidden,
                                        n_obsv=self.n_obsv, n_step=self.n_step,
                                        order=self.order, start=self.start, n_iter=self.n_iter)
        else:
            raise NotImplementedError

        self.params_Estep = self.factor.params_Estep
        self.params_Mstep = self.factor.params_Mstep
        self.L1 = self.factor.L1
        self.L2_sqr = self.factor.L2_sqr

        self.y_pred = self.factor.y_pred
        self.z_pred = self.factor.z_pred
        self.y_next = self.factor.y_next
        self.z_next = self.factor.z_next
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

        # Loss = ||Z*(t)-Z(t)||^2 + ||Y*(t) - Y(t)||^2
        self.z_std = self.z[self.start + self.order: self.start + self.order + self.n_iter]
        self.loss_Estep = lambda y : self.mse(self.y_pred, y) + self.mse(self.z_pred, self.z[self.order:])
        self.loss_Mstep = lambda y : self.mse(self.y_next, y) + self.mse(self.z_next, self.z_std)
        self.test_loss = lambda y : self.mse(self.y_next, y)

        # Smooth Term ||Z(t+1)-Z(t)||^2
        # Estep
        diag_Estep = np.zeros((self.n_step + self.order, self.n_step + self.order),
                                dtype=theano.config.floatX)
        np.fill_diagonal(diag_Estep[:,1:], 1.)
        diag_Estep[-1,-1] = 1.
        z_tm1 = T.dot(diag_Estep, self.z)
        self.smooth_Estep = self.mse(self.z, z_tm1)

        diag_Mstep = T.eye(self.n_iter, self.n_iter, 1)
        diag_Mstep = T.set_subtensor(diag_Mstep[-1, -1], 1)
        z_tm1_next = T.dot(diag_Mstep, self.z_next)
        self.smooth_Mstep = self.mse(self.z_next, z_tm1_next)
    def mse(self, y_1, y_2):
        # error between output and target
        return T.mean((y_1 - y_2) ** 2)


class MetaDFG(BaseEstimator):
    def __init__(self, n_hidden, n_obsv, n_step, order, learning_rate_Estep=0.1, learning_rate_Mstep=0.1,
                n_epochs=100, batch_size=100, L1_reg=0.00, L2_reg=0.00, smooth_reg=0.00,
                learning_rate_decay=1, learning_rate_decay_every=100,
                factor_type='FIR', activation='tanh', final_momentum=0.9,
                initial_momentum=0.5, momentum_switchover=5,
                n_iters=[1,], n_iter_change_every=None, snapshot_every=None, snapshot_path='tmp/'):
        self.n_hidden = int(n_hidden)
        self.n_obsv = int(n_obsv)
        self.n_step = int(n_step)
        self.order = int(order)
        self.learning_rate_Estep = float(learning_rate_Estep)
        self.learning_rate_Mstep = float(learning_rate_Mstep)
        self.learning_rate_decay = float(learning_rate_decay)
        self.learning_rate_decay_every=int(learning_rate_decay_every)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.smooth_reg = float(smooth_reg)
        self.factor_type = factor_type
        self.activation = activation
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)
        self.n_iters = n_iters
        self.n_iter_change_every = n_iter_change_every
        if snapshot_every is not None:
            self.snapshot_every = int(snapshot_every)
        else:
            self.snapshot_every = None
        self.snapshot_path = snapshot_path
        self.ready()

    def ready(self):
        # observation (where first dimension is time)
        self.y = T.matrix(name='y', dtype=theano.config.floatX)

        # learning rate
        self.lr = T.scalar()
        # For mini-batch
        self.start = T.iscalar('start')
        self.n_iter = T.iscalar('n_iter')

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
                        order=self.order, start=self.start,
                        n_iter=self.n_iter, factor_type=self.factor_type)

    def shared_dataset(self, data):
        """ Load the dataset into shared variables """

        shared_data = theano.shared(np.asarray(data,
                                            dtype=theano.config.floatX))
        return shared_data

    def __getstate__(self):
        params = self.get_params() # all the parameters in self.__init__
        weights_E = [p.get_value() for p in self.dfg.params_Estep]
        weights_M = [p.get_value() for p in self.dfg.params_Mstep]
        weights = (weights_E, weights_M)
        state = (params, weights)
        return state

    def _set_weights(self, weights):
        weights_E, weights_M = weights
        i = iter(weights_E)
        for param in self.dfg.params_Estep:
            param.set_value(i.next())
        i = iter(weights_M)
        for param in self.dfg.params_Mstep:
            param.set_value(i.next())

    def __setstate__(self, state):
        params, weights = state
        self.set_params(**params)
        self.ready()
        self._set_weights(weights)

    def save(self, fpath='.', fname=None):
        """Save a pickled representation of model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            fpath, fname = os.path.split(fpath)
        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)
        logger.info('Saving to %s ...' % fabspath)
        with open(fabspath, 'wb') as file:
            state = self.__getstate__()
            pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, fpath):
        """ Load model parameters from fpath. """
        logger.info('Loading from %s ...' % fpath)
        with open(fpath, 'rb') as file:
            state = pickle.load(file)
            self.__setstete__(state)

    def fit(self, Y_train, Y_test=None,
            validation_frequency=100):
        """Fit model

        Pass in Y_test to compute test error and report during training
            Y_train : ndarray (n_seq, n_step, n_out)
            Y_test  : ndarray (n_seq, n_step, n_out)

        validation_frequency : int
            in terms of number of epoch
        """


        if Y_test is not None:
            self.interactive = True
            test_set_y = self.shared_dataset(Y_test)
        else:
            self.interactive = False
        train_set_y = self.shared_dataset(Y_train)
        n_train = train_set_y.get_value(borrow=True).shape[0]
        if self.interactive:
            n_test = test_set_y.get_value(borrow=True).shape[0]

        logger.info('...building the model')

        index = T.lscalar('index') #index to a case
        # learning rate (may change)
        l_r = T.scalar('l_r', dtype=theano.config.floatX)
        mom = T.scalar('mom', dtype=theano.config.floatX)

        cost_Estep = self.dfg.loss_Estep(self.y) \
                + self.dfg.smooth_Estep \
                + self.L1_reg * self.dfg.L1 \
                + self.L2_reg * self.dfg.L2_sqr

        cost_Mstep = self.dfg.loss_Mstep(self.y) \
                + self.dfg.smooth_Mstep \
                + self.L1_reg * self.dfg.L1 \
                + self.L2_reg * self.dfg.L2_sqr

        # compute the gradient of cost with respect to theta = (W, W_in, W_out)
        # gradients on the weights using BPTT
        # E step
        gparams_Estep = []
        for param in self.dfg.params_Estep:
            gparam = T.grad(cost_Estep, param)
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
            gparam = T.grad(cost_Mstep, param)
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
                                        outputs=[cost_Estep, self.dfg.y_pred, self.dfg.z_pred],
                                        updates=updates_Estep,
                                        givens=OrderedDict([(self.y, train_set_y[index])]),
                                        mode=mode)
        # updates the parameter of the model based on
        # the rules defined in `updates_Mstep`
        train_model_Mstep = theano.function(inputs=[index, l_r, mom, self.start, self.n_iter],
                                        outputs=[cost_Mstep, self.dfg.y_next, self.dfg.z_next],
                                        updates=updates_Mstep,
                                        givens=OrderedDict([(self.y, train_set_y[index][self.start: self.start + self.n_iter])]),
                                        mode=mode)
        test_model = theano.function(inputs=[self.start, self.n_iter],
                                    outputs=[self.dfg.y_next, self.dfg.z_next],
                                    #givens=OrderedDict([(self.y, test_set_y[index])]),
                                    mode=mode)
        ###############
        # TRAIN MODEL #
        ###############
        logger.info('... training')
        epoch = 0
        n_iter = self.n_iters[0]
        self.n_iters = self.n_iters[1:]

        while (epoch < self.n_epochs):
            epoch = epoch + 1
            average_cost = 0.
            for idx in xrange(n_train):
                effective_momentum = self.final_momentum \
                               if epoch > self.momentum_switchover \
                               else self.initial_momentum
                example_cost, example_y_pred, example_z_pred = train_model_Estep(idx, self.learning_rate_Estep,
                                           effective_momentum)
                average_cost += example_cost
            logger.info('epoch %d E_step cost=%f' % (epoch, average_cost / n_train))
            average_cost = []
            for idx in xrange(n_train):
                for head in xrange(0, self.n_step - n_iter):
                    effective_momentum = self.final_momentum \
                                if epoch > self.momentum_switchover \
                                else self.initial_momentum

                    example_cost, example_y_next, example_z_next = train_model_Mstep(idx, self.learning_rate_Mstep,
                                                effective_momentum, head, n_iter)
                    average_cost.append(example_cost)
            logger.info('epoch %d M_step n_iter=%d cost=%f' % (epoch, n_iter, np.mean(average_cost)))
            # Update learning rate
            if self.learning_rate_decay_every is not None:
                if epoch % self.learning_rate_decay_every == 0:
                    self.learning_rate_Estep *= self.learning_rate_decay
                    self.learning_rate_Mstep *= self.learning_rate_decay
            # Update n_iter
            if self.n_iter_change_every is not None:
                if epoch % self.n_iter_change_every == 0:
                    n_iter = self.n_iters[0]
                    if len(self.n_iters) > 1:
                        self.n_iters = self.n_iters[1:]
            # Snapshot
            if self.snapshot_every is not None:
                if epoch % self.snapshot_every == 0:
                    date_obj = datetime.datetime.now()
                    date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
                    class_name = self.__class__.__name__
                    fname = '%s.%s-snapshot-%d.png' % (class_name, date_str, epoch)
                    plt.figure()
                    n = Y_train[0].shape[0] + Y_test[0].shape[0]
                    x = np.linspace(0, n, n)
                    len_train = Y_train[0].shape[0]
                    x_train, x_test = x[:len_train], x[len_train:]
                    plt.plot(x_train, np.squeeze(Y_train[0]), 'b', linewidth=2)
                    plt.plot(x_train, np.squeeze(example_y_pred), 'r', linewidth=2)
                    plt.savefig(self.snapshot_path + fname)
                    plt.close()
                    if self.interactive:
                        y_test_next, z_test_next = test_model(self.n_step, Y_test[0].shape[0])
                        #logger.info('epoch %d test loss=%f' % (epoch, test_loss))
                        plt.figure()
                        plt.plot(x_test, np.squeeze(Y_test[0]), 'b', linewidth=2)
                        plt.plot(x_test, np.squeeze(y_test_next), 'r', linewidth=2)
                        fname = '%s.%s-snapshot-%d_test.png' % (class_name, date_str, epoch)
                        plt.ylim(-5, 5)
                        plt.savefig(self.snapshot_path + fname)
                        plt.close()


class sinTestCase(unittest.TestCase):
    def runTest(self):
        n = 500
        x = np.linspace(0, n, n)
        sita = [.2, .331, .42, .51, .74]
        y = np.zeros(n)
        for item in sita:
            y += np.sin(item * x)
        # n_seq x n_t x n_in
        n_train = 450
        n_test = 50
        y_train = y[:n_train]
        y_test = y[n_train:]
        y_train = y_train.reshape(1, n_train, 1)
        y_test = y_test.reshape(1, n_test, 1)
        dfg = MetaDFG(n_hidden=5, n_obsv=1, n_step=n_train, order=25, learning_rate_Estep=0.01, learning_rate_Mstep=0.01,
                n_epochs=1000, snapshot_every=10, L1_reg=0.01, L2_reg=0.01, smooth_reg=0.01,
                learning_rate_decay=.1, learning_rate_decay_every=500,
                n_iters=[2, 3, 5, 10, 20], n_iter_change_every=30)
        dfg.fit(y_train, y_test)
        assert True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

