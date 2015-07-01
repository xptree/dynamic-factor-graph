#!/usr/bin/env python
# encoding: utf-8
# File Name: Exp.py
# Author: Jiezhong Qiu
# Create Time: 2015/06/23 17:04
# TODO:

from dfg_minibatch import MetaDFG
import Config as config
import cPickle as pickle
import datetime
import logging

logger = logging.getLogger(__name__)

def run(Dir):
    with open(config.getPklDir(), 'rb') as f:
        Y, X, user_id, T = pickle.load(f)
    print X.shape, Y.shape
    n_in = X.shape[2]
    Y_train = Y[:T]
    Y_test = Y[T:]
    X_train = X[:T]
    X_test = X[T:]
    n_step, n_seq, n_obsv = Y_train.shape
    logger.info('load from pkl train_step=%d test_step=%d, n_seq=%d n_obsv=%d n_in=%d', n_step, X_test.shape[0], n_seq, n_obsv, n_in)
    start = datetime.datetime.now()
    dfg = MetaDFG(n_in=n_in, n_hidden=2, n_obsv=n_obsv, n_step=n_step, order=2, n_seq=n_seq, learning_rate_Estep=0.5, learning_rate_Mstep=0.1,
            factor_type='MLP', output_type='binary',
            n_epochs=100, batch_size=n_seq , snapshot_every=None, L1_reg=0.00, L2_reg=0.00, smooth_reg=0.00,
            learning_rate_decay=.5, learning_rate_decay_every=100,
            n_iter_low=[n_step / 2] , n_iter_high=[n_step + 1], n_iter_change_every=100,
            final_momentum=0.5,
            initial_momentum=0.3, momentum_switchover=1500,
            order_obsv=0,
            hidden_layer_config=[])
    #X_train = np.zeros((n_step, n_seq, n_in))
    #X_test = np.zeros((Y_test.shape[0], n_seq, n_in))
    cert_pred = dfg.fit(Y_train=Y_train, X_train=X_train, Y_test=Y_test, X_test=X_test, validation_frequency=None)
    with open(Dir, 'wb') as f:
        for i in xrange(len(user_id)):
            print >> f, '\t'.join([str(user_id[i]), str(cert_pred[i])])
    print datetime.datetime.now() - start


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s') # include timestamp
    run(config.getPredictionResultDir())
