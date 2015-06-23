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
import json
import datetime
import os
import cPickle as pickle
import factor_minibatch
import unittest
#import matplotlib.pylab as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score


logger = logging.getLogger(__name__)
theano.config.exception_verbosity='high'
mode = theano.Mode(linker='cvm')
#mode = 'DebugMode'
#mode = 'FAST_COMPILE'
#mode = theano.Mode(optimizer=None)
#mode = 'ProfileMode'
DEBUG = True
THRESHOLD = (1 - 41./513) * 100
WEIGHT = [1./28] * 8 + [5. / 7]
DATA_DIR = 'data/circuit.pkl'
THRESHOLD = 70.3
WEIGHT = [5.] * 10 + [50.]
DATA_DIR = 'data/fin2.pkl'

def getBestThreshold(yres, ystd):
    yList = sorted([ (yres[i], ystd[i]) for i in xrange(len(yres)) ], key = lambda item: item[0])
    tn = tp = fn = fp = 0.
    Bestf1 = 0.
    for y_, y in yList:
        if y == 1:
            tp += 1
        else:
            fn += 1
    for item in yList:
        #this item predicted as negtive
        if item[1] == 1:
            fp += 1
            tp -= 1
        else:
            tn += 1
            fn -= 1
        if tp + fn == 0 or tp + fp == 0:
            continue
        prec = tp / (tp + fn)
        rec = tp / (tp + fp)
        if prec + rec < 1e-5:
            continue
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > Bestf1:
            Bestf1 = f1
            threshold = item[0]
    return threshold
def metric(y_pred, y_std):
    threshold = getBestThreshold(y_pred, y_std)
    prf = precision_recall_fscore_support(y_std, y_pred > threshold, average='micro')
    return 'auc %f, prec %f, rec %f, f1 %f' % (roc_auc_score(y_std, y_pred), prf[0], prf[1], prf[2])

class DFG(object):
    """     Dynamic factor graph class

    Support output types:
    real: linear output units, use mean-squared error
    binary: binary output units, use cross-entropy error
    softmax: single softmax out, use cross-entropy error
    """
    def __init__(self, n_in, x, y_pad, n_hidden, n_obsv, n_step, order, n_seq, start, n_iter,
                factor_type='FIR', output_type='real',
                order_obsv=0, hidden_layer_config=None):
        self.n_in = n_in
        self.x = x
        self.y_pad = y_pad
        self.n_hidden = n_hidden
        self.n_obsv = n_obsv
        self.n_step = n_step
        self.order = order
        self.n_seq = n_seq
        self.factor_type = factor_type
        self.start = start
        self.n_iter = n_iter
        # For mini-batch
        self.index = T.iscalar('index') # index to a [mini]batch
        self.n_ex  = T.iscalar('n_ex') # the number of examples
        self.batch_size = T.iscalar('batch_size')
        self.batch_start = self.index * self.batch_size
        self.batch_stop = T.minimum(self.n_ex, (self.index + 1) * self.batch_size)
        self.effective_batch_size = self.batch_stop - self.batch_start
        self.order_obsv=order_obsv
        self.hidden_layer_config = hidden_layer_config
        if self.factor_type == 'FIR':
            # FIR factor with n_in > 0 is not implemented
            if self.n_in > 0:
                raise NotImplementedError
            self.factor = factor_minibatch.FIR(n_hidden=self.n_hidden,
                                        n_obsv=self.n_obsv, n_step=self.n_step,
                                        order=self.order, n_seq=self.n_seq, start=self.start, n_iter=self.n_iter,
                                        batch_start=self.batch_start, batch_stop=self.batch_stop)
        elif self.factor_type == 'MLP':
            self.factor = factor_minibatch.MLP(n_in=self.n_in,
                                        x=self.x, y_pad=self.y_pad,
                                        n_hidden=self.n_hidden,
                                        n_obsv=self.n_obsv, n_step=self.n_step,
                                        order=self.order, n_seq=self.n_seq, start=self.start, n_iter=self.n_iter,
                                        batch_start=self.batch_start, batch_stop=self.batch_stop,
                                        order_obsv=self.order_obsv,
                                        hidden_layer_config=self.hidden_layer_config)
        else:
            raise NotImplementedError
        self.output_type = output_type

        self.params_Estep = self.factor.params_Estep
        self.params_Mstep = self.factor.params_Mstep
        self.L1 = self.factor.L1
        self.L2_sqr = self.factor.L2_sqr

        self.y_pred_Estep = self.factor.y_pred_Estep
        self.z_pred_Estep = self.factor.z_pred_Estep
        self.y_pred_Mstep = self.factor.y_pred_Mstep
        self.z_pred_Mstep = self.factor.z_pred_Mstep
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
        self.z_std = self.z[self.start+self.order:self.start+self.order+self.n_iter,self.batch_start:self.batch_stop]
        if self.output_type == 'real':
            self.loss_Estep = lambda y : (self.se(self.y_pred_Estep, y) + self.se(self.z_pred_Estep, self.z[self.order:])) / n_seq
            self.loss_Mstep = lambda y : (self.se(self.y_pred_Mstep, y) + self.se(self.z_pred_Mstep, self.z_std)) / self.effective_batch_size
            self.test_loss = lambda y : self.se(self.y_next, y) / self.effective_batch_size
        elif self.output_type == 'binary':
            self.loss_Estep = lambda y : (self.nll_binary(self.y_pred_Estep, y) \
                                + self.se(self.z_pred_Estep, self.z[self.order:]) \
                                + 0 * self.nll_binary(self.y_pred_Estep[:,:,-1], y[:,:,-1])) / n_seq
            self.loss_Mstep = lambda y : (self.nll_binary(self.y_pred_Mstep, y) \
                                + self.se(self.z_pred_Mstep, self.z_std) \
                                + 0 * self.nll_binary(self.y_pred_Mstep[:,:,-1], y[:,:,-1])) / self.effective_batch_size
            self.test_loss = lambda y : self.nll_binary(self.y_next, y) / self.effective_batch_size
        else:
            raise NotImplementedError

        # Smooth Term ||Z(t+1)-Z(t)||^2
        # Estep
        diag_Estep = np.zeros(((n_step + order)*n_hidden, (n_step+order)*n_hidden),
                                dtype=theano.config.floatX)
        np.fill_diagonal(diag_Estep[n_hidden:,:], 1.)
        np.fill_diagonal(diag_Estep[-n_hidden:,-n_hidden:], 1.)
        # (n_step+order) x n_seq x n_hdden
        z_flatten = T.flatten(self.z.dimshuffle(1, 0, 2), outdim=2)
        z_tm1 = T.dot(z_flatten, diag_Estep)
        self.smooth_Estep = self.se(z_flatten, z_tm1) / n_seq

        diag_Mstep = T.eye(self.n_iter*n_hidden, self.n_iter*n_hidden, n_hidden)
        for i in xrange(n_hidden):
            diag_Mstep = T.set_subtensor(diag_Mstep[-i-1, -i-1], 1)
        z_next_flatten = T.flatten(self.z_next.dimshuffle(1, 0, 2), outdim=2)
        z_next_tm1 = T.dot(z_next_flatten, diag_Mstep)
        self.smooth_Mstep = self.se(z_next_flatten, z_next_tm1) / self.effective_batch_size
    def se(self, y_1, y_2):
        return T.sum((y_1 - y_2) ** 2)
    def mse(self, y_1, y_2):
        # error between output and target
        return T.mean((y_1 - y_2) ** 2)
    def nmse(self, y_1, y_2):
        # treat y_1 as the approximation to y_2
        return self.mse(y_1, y_2) / self.mse(y_2, 0)
    def nll_binary(self, y_1, y_2):
        return T.sum(T.nnet.binary_crossentropy(y_1, y_2))

    def prec(self, y, y_pred):
        y_out = T.round(y_pred[-1,:,-1])
        y_std = T.round(y[-1,:,-1])
        true_pos = T.sum(T.eq(y_std, 1) * T.eq(y_out, 1))
        false_pos = T.sum(T.neq(y_std, 1) * T.eq(y_out, 1))
        return (true_pos + 0.) / (true_pos + false_pos)
    def rec(self, y, y_pred):
        y_out = T.round(y_pred[-1,:,-1])
        y_std = T.round(y[-1,:,-1])
        true_pos = T.sum(T.eq(y_std, 1) * T.eq(y_out, 1))
        false_neg = T.sum(T.eq(y_std, 1) * T.neq(y_out, 1))
        return (true_pos + 0.) / (true_pos + false_neg)

class MetaDFG(BaseEstimator):
    def __init__(self, n_in, n_hidden, n_obsv, n_step, order, n_seq, learning_rate_Estep=0.1, learning_rate_Mstep=0.1,
                n_epochs=100, batch_size=100, L1_reg=0.00, L2_reg=0.00, smooth_reg=0.00,
                learning_rate_decay=1, learning_rate_decay_every=100,
                factor_type='FIR', output_type='real', activation='tanh', final_momentum=0.9,
                initial_momentum=0.5, momentum_switchover=5,
                n_iter_low=[20,], n_iter_high=[50,], n_iter_change_every=50,
                snapshot_every=None, snapshot_path='tmp/',
                order_obsv=0,
                hidden_layer_config=None):
        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.n_obsv = int(n_obsv)
        self.n_step = int(n_step)
        self.order = int(order)
        self.n_seq = int(n_seq)
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
        self.output_type = output_type
        self.activation = activation
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)
        self.n_iter_low = n_iter_low
        self.n_iter_high = n_iter_high
        assert(len(self.n_iter_low) == len(self.n_iter_high))
        self.n_iter_change_every = int(n_iter_change_every)
        if snapshot_every is not None:
            self.snapshot_every = int(snapshot_every)
        else:
            self.snapshot_every = None
        self.snapshot_path = snapshot_path
        self.order_obsv = int(order_obsv)
        self.hidden_layer_config = hidden_layer_config
        self.ready()

    def ready(self):
        # observation (where first dimension is time)
        self.y = T.tensor3(name='y', dtype=theano.config.floatX)
        self.y_pad = T.tensor3(name='y_pad', dtype=theano.config.floatX)
        self.x = T.tensor3(name='x', dtype=theano.config.floatX)

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

        self.dfg = DFG(n_in=self.n_in, x=self.x, y_pad=self.y_pad, n_hidden=self.n_hidden,
                        n_obsv=self.n_obsv, n_step=self.n_step,
                        order=self.order, n_seq=self.n_seq, start=self.start,
                        n_iter=self.n_iter, factor_type=self.factor_type,
                        output_type=self.output_type,
                        order_obsv=self.order_obsv,
                        hidden_layer_config=self.hidden_layer_config)

    def shared_dataset(self, data):
        """ Load the dataset into shared variables """

        shared_data = theano.shared(np.asarray(data,
                                            dtype=theano.config.floatX))
        return shared_data

    def __getstate__(self, jsonobj=False):
        params = self.get_params() # all the parameters in self.__init__
        weights_E = [p.get_value().tolist() if jsonobj else p.get_value() for p in self.dfg.params_Estep]
        weights_M = [p.get_value().tolist() if jsonobj else p.get_value() for p in self.dfg.params_Mstep]
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
        elif fpathext == '.json':
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
            if fpathext == '.json':
                state = self.__getstate__(jsonobj=True)
                json.dump(state, file,
                            indent=4, separators=(',', ': '))
            else:
                state = self.__getstate__()
                pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, fpath):
        """ Load model parameters from fpath. """
        logger.info('Loading from %s ...' % fpath)
        with open(fpath, 'rb') as file:
            state = pickle.load(file)
            self.__setstete__(state)

    def fit(self, Y_train=None, Y_test=None,
            X_train=None, X_test=None,
            validation_frequency=100):
        """Fit model

        Pass in Y_test to compute test error and report during training
            Y_train : ndarray (n_step, n_seq, n_out)
            Y_test  : ndarray (T, n_seq, n_out)
            X_train : ndarray (n_step, n_seq, n_in)
            X_test  : ndarray (T, n_seq, n_in)
        validation_frequency : int
            in terms of number of epoch
        """


        if Y_test is not None:
            self.interactive = True
            test_set_y = self.shared_dataset(Y_test)
            test_set_x = self.shared_dataset(X_test)
            Y_test_binary = np.zeros_like(Y_test,
                                    dtype=theano.config.floatX)
            for t in xrange(Y_test.shape[0]):
                for i in xrange(self.n_obsv):
                    threshold = np.percentile(Y_test[t,:,i], THRESHOLD)
                    Y_test_binary[t,:,i] = Y_test[t,:,i] >= threshold
            test_set_y_binary = self.shared_dataset(Y_test_binary)
        else:
            self.interactive = False

        train_set_x = self.shared_dataset(X_train)
        # generate Y_pad
        Y_train_pad = np.zeros(shape=(self.n_step + self.order_obsv, self.n_seq, self.n_obsv),
                                dtype=theano.config.floatX)
        Y_train_pad[self.order_obsv:,:,:]=Y_train
        # generate Y_binary
        Y_train_binary = np.zeros_like(Y_train,
                                dtype=theano.config.floatX)
        for t in xrange(self.n_step):
            for i in xrange(self.n_obsv):
                threshold = np.percentile(Y_train[t,:,i], THRESHOLD)
                Y_train_binary[t,:,i] = Y_train[t,:,i] >= threshold
        train_set_y = self.shared_dataset(Y_train)
        train_set_y_pad = self.shared_dataset(Y_train_pad)
        train_set_y_binary = self.shared_dataset(Y_train_binary)
        n_train = train_set_y.get_value(borrow=True).shape[1]
        n_train_batches = int(np.ceil(float(n_train) / self.batch_size))
        if self.interactive:
            n_test = test_set_y.get_value(borrow=True).shape[1]
            n_test_batches = int(np.ceil(float(n_test) / self.batch_size))

        logger.info('...building the model')

        index = self.dfg.index
        n_ex = self.dfg.n_ex
        # learning rate (may change)
        l_r = T.scalar('l_r', dtype=theano.config.floatX)
        mom = T.scalar('mom', dtype=theano.config.floatX)

        cost_Estep = self.dfg.loss_Estep(self.y) \
                + self.smooth_reg * self.dfg.smooth_Estep \
                + self.L1_reg * self.dfg.L1 \
                + self.L2_reg * self.dfg.L2_sqr

        cost_Mstep = self.dfg.loss_Mstep(self.y) \
                + self.smooth_reg * self.dfg.smooth_Mstep \
                + self.L1_reg * self.dfg.L1 \
                + self.L2_reg * self.dfg.L2_sqr

        # mini-batch implement
        batch_size = self.dfg.batch_size
        batch_start = self.dfg.batch_start
        batch_stop = self.dfg.batch_stop
        effective_batch_size = self.dfg.effective_batch_size
        get_batch_size = theano.function(inputs=[index, n_ex, batch_size],
                                            outputs=effective_batch_size)

        givens=[(self.y, train_set_y_binary),
                (self.x, train_set_x),
                (self.y_pad, train_set_y_pad)]
        if self.order_obsv == 0:
            givens = givens[:-1]
        compute_train_error_Estep = theano.function(inputs=[],
                                                outputs=[self.dfg.loss_Estep(self.y), self.dfg.y_pred_Estep,
                                                            self.dfg.prec(self.y, self.dfg.y_pred_Estep), self.dfg.rec(self.y, self.dfg.y_pred_Estep)],
                                                givens=OrderedDict(givens),
                                                mode=mode)

#        compute_train_error_Mstep = theano.function(inputs=[index, n_ex, self.start, self.n_iter, batch_size],
#                                        outputs=self.dfg.loss_Mstep(self.y),
#                                        givens=OrderedDict([(self.y, train_set_y[self.start:self.start+self.n_iter, batch_start:batch_stop]),
#                                                            (self.y_pad, train_set_y[self.start:self.start+1, batch_start:batch_stop]),
#                                                            (self.x, train_set_x[self.start:self.start+self.n_iter, batch_start:batch_stop])]),
#                                        mode=mode)
        if self.interactive:
            givens=[(self.y, test_set_y_binary[:, batch_start:batch_stop]),
                    (self.x, test_set_x[:,batch_start:batch_stop]),
                    (self.y_pad, train_set_y_pad)]
            if self.order_obsv == 0:
                givens = givens[:-1]
            compute_test_error = theano.function(inputs=[index, n_ex, self.start, self.n_iter, batch_size],
                                                    outputs=[self.dfg.test_loss(self.y), self.dfg.y_next, self.y],
                                                    givens=OrderedDict(givens),
                                                    mode=mode)


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
        givens=[(self.y, train_set_y_binary),
                (self.x, train_set_x),
                (self.y_pad, train_set_y_pad)]
        if self.order_obsv == 0:
            givens = givens[:-1]
        train_model_Estep = theano.function(inputs=[l_r, mom],
                                        outputs=[cost_Estep, self.dfg.loss_Estep(self.y), self.dfg.y_pred_Estep, self.dfg.z_pred_Estep],
                                        updates=updates_Estep,
                                        givens=OrderedDict(givens),
                                        mode=mode)
        # updates the parameter of the model based on
        # the rules defined in `updates_Mstep`
        givens=[(self.y, train_set_y_binary[self.start:self.start+self.n_iter, batch_start:batch_stop]),
                (self.x, train_set_x[self.start:self.start+self.n_iter, batch_start:batch_stop]),
                (self.y_pad, train_set_y_pad) ]
        if self.order_obsv == 0:
            givens = givens[:-1]
        train_model_Mstep = theano.function(inputs=[index, n_ex, l_r, mom, self.start, self.n_iter, batch_size],
                                        outputs=[cost_Mstep, self.dfg.y_pred_Mstep, self.dfg.z_pred_Mstep],
                                        updates=updates_Mstep,
                                        givens=OrderedDict(givens),
                                        mode=mode)
        ###############
        # TRAIN MODEL #
        ###############
        logger.info('... training')
        epoch = 0
        auc = []
        while (epoch < self.n_epochs):
            epoch = epoch + 1
            effective_momentum = self.final_momentum \
                        if epoch > self.momentum_switchover \
                        else self.initial_momentum
            example_cost, example_energy, example_y_pred_Estep, example_z_pred_Estep = train_model_Estep(self.learning_rate_Estep, 0.)
            logger.info('epoch %d E_step cost=%f energy=%f' % (epoch,
                                        example_cost, example_energy))
            for minibatch_idx in xrange(n_train_batches):
                average_cost = []
                for i in xrange(self.n_step):
                    n_iter = np.random.randint(low=self.n_iter_low[0],
                                                high=self.n_iter_high[0])
                    head = np.random.randint(self.n_step - n_iter + 1)
                    example_cost, example_y_pred_Mstep, example_z_pred_Mstep = train_model_Mstep(minibatch_idx, n_train, self.learning_rate_Mstep,
                                                effective_momentum, head, n_iter, self.batch_size)
                    average_cost.append(example_cost)
                    '''
                    test_losses, test_precs, test_recs = [], [], []
                    auc_now = []
                    for ii in xrange(n_test_batches):
                        test_loss, y_next, y_std = compute_test_error(ii, n_test, self.n_step, Y_test.shape[0], self.batch_size)
                        for j in xrange(y_next.shape[0]):
                            auc_now.append(roc_auc_score(y_std[j,:,-1], y_next[j,:,-1]))
                    auc.append(np.mean(auc_now))
                    '''
                logger.info('epoch %d batch %d M_step cost=%f' % (epoch, minibatch_idx, np.mean(average_cost)))
                #iters = (epoch - 1) * n_train_batches + minibatch_idx + 1
            if epoch % validation_frequency == 0:
                # Computer loss on training set (conside Estep loss only)
                train_loss_Estep, y_pred_Estep, prec, rec = compute_train_error_Estep()
                #print np.max(y_pred_Estep), np.min(y_pred_Estep)
                if self.interactive:
                    test_losses, test_precs, test_recs = [], [], []
                    for i in xrange(n_test_batches):
                        test_loss, y_next, y_std = compute_test_error(i, n_test, self.n_step, Y_test.shape[0], self.batch_size)
                        logger.info('epoch %d, %s batch tr_loss %f te_loss %f' % \
                                    (epoch, 'valid' if i==0 else 'test', train_loss_Estep, test_loss))
                        for j in xrange(y_next.shape[0]):
                            logger.info('behavior at time stamp %d' % (self.n_step + j + 1))
                            logger.info('%s' % \
                                        metric(y_next[j,:,-1], y_std[j,:,-1]))
                            #logger.info('train_max %f train_min %f test_max %f test_min %f' % \
                        #            (np.max(y_pred_Estep[-1,:,-1]), np.min(y_pred_Estep[-1,:,-1]), np.max(_[-1,:,-1]), np.min(_[-1,:,-1])))
                        if i == 0:
                            y_historic = Y_train[:,:self.batch_size,:]
                        else:
                            y_historic = Y_train[:,self.batch_size:,:]
                        y_std = np.concatenate([y_historic, y_std])
                        y_next = np.concatenate([y_historic, y_next])
                        cert_pred = np.squeeze(np.average(y_next, axis=0, weights=WEIGHT))
                        cert_std = np.squeeze(np.average(y_std, axis=0, weights=WEIGHT))
                        median = np.percentile(cert_std, THRESHOLD)
                        cert_std = cert_std > median
                        logger.info('certificate prediction')
                        logger.info('%s' % \
                                metric(cert_pred, cert_std))
                else:
                    logger.info('epoch %d, tr_loss %f tr_prec %f tr_rec %f' % \
                                (epoch, train_loss_Estep, prec, rec))
            # Update learning rate
            if self.learning_rate_decay_every is not None:
                if epoch % self.learning_rate_decay_every == 0:
                    self.learning_rate_Estep *= self.learning_rate_decay
                    self.learning_rate_Mstep *= self.learning_rate_decay
            if epoch % self.n_iter_change_every == 0:
                if len(self.n_iter_low) > 1:
                    self.n_iter_low = self.n_iter_low[1:]
                    self.n_iter_high = self.n_iter_high[1:]
            '''
            # Snapshot
            if self.snapshot_every is not None:
                if (epoch - 1) % self.snapshot_every == 0:
                    date_obj = datetime.datetime.now()
                    date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
                    class_name = self.__class__.__name__
                    fname = '%s.%s-snapshot-%d.png' % (class_name, date_str, epoch)
                    plt.figure()
                    n = Y_train.shape[0] + Y_test.shape[0]
                    x = np.linspace(0, n, n)
                    len_train = Y_train.shape[0]
                    x_train, x_test = x[:len_train], x[len_train:]
                    plt.plot(x_train, np.squeeze(Y_train), 'b', linewidth=2)
                    plt.plot(x_train, np.squeeze(example_y_pred_Estep), 'r', linewidth=2)
                    plt.savefig(self.snapshot_path + fname)
                    plt.close()
                    if self.interactive:
                        y_test_next = compute_test_error(0, n_test, self.n_step, Y_test.shape[0], self.batch_size)[1]
                        #logger.info('epoch %d test loss=%f' % (epoch, test_loss))
                        plt.figure()
                        plt.plot(x_test, np.squeeze(Y_test), 'b', linewidth=2)
                        plt.plot(x_test, np.squeeze(y_test_next), 'r', linewidth=2)
                        fname = '%s.%s-snapshot-%d_test.png' % (class_name, date_str, epoch)
                        plt.ylim(-3, 3)
                        plt.savefig(self.snapshot_path + fname)
                        plt.close()
            '''
            # Snapshot
            if self.snapshot_every is not None:
                if (epoch + 1) % self.snapshot_every == 0:
                    date_obj = datetime.datetime.now()
                    date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
                    class_name = self.__class__.__name__
                    fname = '%s.%s-snapshot-%d.pkl' % (class_name, date_str, epoch + 1)
                    fname = '%s.%s-snapshot-%d.json' % (class_name, date_str, epoch + 1)
                    fabspath = os.path.join(self.snapshot_path, fname)
                    self.save(fpath=fabspath)
        with open('auc.json', 'wb') as f:
            json.dump(auc, f,
                        indent=4, separators=(',', ': '))

'''
class sinTestCase(unittest.TestCase):

    def runTest(self):
        n = 2500
        x = np.linspace(0, n, n)
        sita = [.2, .331, .42, .51, .74]
        sita = sita[:3]
        y = np.zeros(n)
        for item in sita:
            y += np.sin(item * x)
        # n_t x n_seq x n_in
        n_train = n - 500
        n_test = 500
        y_train = y[:n_train]
        y_test = y[n_train:]
        y_train = y_train.reshape(n_train, 1, 1)
        y_test = y_test.reshape(n_test, 1, 1)
        dfg = MetaDFG(n_hidden=3, n_obsv=1, n_step=n_train, order=25, n_seq=1, learning_rate_Estep=0.01, learning_rate_Mstep=0.001,
                n_epochs=1000, batch_size=1, snapshot_every=1, L1_reg=0.02, L2_reg=0.02, smooth_reg=0.01,
                learning_rate_decay=.9, learning_rate_decay_every=50,
                n_iter_low=[20, 20, 20, 20] , n_iter_high=[31, 51, 71, 101], n_iter_change_every=15,
                final_momentum=0.9,
                initial_momentum=0.5, momentum_switchover=500)
        dfg.fit(y_train, y_test, validation_frequency=1)
        assert True
'''
class xtxTestCase(unittest.TestCase):
    def runTest(self):
        with open(DATA_DIR, 'rb') as file:
            Y, X = pickle.load(file)
        #X = X[:,:1000,:]
        #Y = Y[:,:1000,:]
        n_in = X.shape[2]
        T = -6
        #T = -9
        Y_train = Y[:T]
        Y_test = Y[T:]
        X_train = X[:T]
        X_test = X[T:]
        #print np.sum(data[-1,:,-1])
        n_step, n_seq, n_obsv = Y_train.shape
        logger.info('load from pkl train_step=%d test_step=%d, n_seq=%d n_obsv=%d n_in=%d', n_step, X_test.shape[0], n_seq, n_obsv, n_in)
        start = datetime.datetime.now()
        dfg = MetaDFG(n_in=n_in, n_hidden=2, n_obsv=n_obsv, n_step=n_step, order=2, n_seq=n_seq, learning_rate_Estep=0.5, learning_rate_Mstep=0.1,
                factor_type='MLP', output_type='binary',
                n_epochs=2000, batch_size=n_seq , snapshot_every=1000, L1_reg=0.00, L2_reg=0.00, smooth_reg=0.00,
                learning_rate_decay=.5, learning_rate_decay_every=100,
                n_iter_low=[n_step / 2] , n_iter_high=[n_step + 1], n_iter_change_every=100,
                final_momentum=0.5,
                initial_momentum=0.3, momentum_switchover=1500,
                order_obsv=0,
                hidden_layer_config=[])
        #X_train = np.zeros((n_step, n_seq, n_in))
        #X_test = np.zeros((Y_test.shape[0], n_seq, n_in))
        dfg.fit(Y_train=Y_train, X_train=X_train, Y_test=Y_test, X_test=X_test, validation_frequency=2000)
        print datetime.datetime.now() - start

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s') # include timestamp
    unittest.main()


