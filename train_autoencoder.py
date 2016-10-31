#! /usr/bin/python
import os, sys
import numpy as np
import tensorflow as tf
import cPickle
from IPython import embed
import datetime as dttm
from datetime import date
import matplotlib.pyplot as plt
import keras.backend as K
import pandas as pd
import argparse

from model_builder import build_autoencoder
from keras.datasets import mnist

class Empty(object):
    def __init__(self,name='duck'):
        self.name = name


def load_data():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = train_x / 255.
    test_x = test_x / 255.
    train_x = train_x.reshape((len(train_x),np.prod(train_x.shape[1:])))
    test_x = test_x.reshape((len(test_x),np.prod(test_x.shape[1:])))
    return (train_x, train_y), (test_x, test_y)


def display_images(original_imgs, decoded_imgs, encoded_imgs):
    n = 10
    plt.figure(figsize=(20,4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(original_imgs[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def predict_model(sess, model, x):
    feed_dict = {
            model.input: x,
            K.learning_phase(): 0}
    pred_vals = sess.run(model.output, feed_dict=feed_dict)   
    return pred_vals


def train_model(train_data, valid_data, p):
    (train_x, train_y), (valid_x, valid_y) = train_data, test_data
    lr = p.init_lr

    in_shape = (None, train_x.shape[1])
    model, encoder_model, decoder_model = build_autoencoder(in_shape, p.l2reg)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)

    tfepoch = tf.Variable(0)
    saver = tf.train.Saver()

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    print 'Training Starts ...\n'
    for epoch in range(p.num_epochs):
        # evaluate on valid data
        valid_dict = {
                model.input: valid_x,
                K.learning_phase(): 0}
        valid_loss = sess.run(model.loss, feed_dict=valid_dict)
        if epoch ==0:
            print '      Epoch 0: valid_loss = %.6f'%valid_loss
        elif epoch % p.display_interval == 0:
            print '      Epoch %d:, valid_loss = %.6f'%(epoch, valid_loss)
        
        # train on train data
        start_idx = 0
        train_loss = []
        while start_idx < len(train_x):
            train_dict = {
                    model.input: train_x[start_idx:start_idx + p.batch_size],
                    learning_rate: lr,
                    K.learning_phase(): 1}
            _, train_loss_ = sess.run([train_step, model.loss],
                    feed_dict=train_dict)
            start_idx += p.batch_size
            train_loss.append(train_loss_)
        train_loss = np.mean(train_loss)
        if epoch % p.display_interval == 0:
            print 'Epoch %d:, train_loss = %.6f'%(epoch, train_loss)
        lr = p.init_lr * (1+p.gamma*epoch)**(-p.power)
        
    #save model
    epochassign = tfepoch.assign(epoch)
    sess.run(epochassign)
    if not p.modelfn is None:
        save_model(sess, saver,modelfn)
    
    # generate decoded images to show
    encoded_imgs = predict_model(sess, encoder_model, valid_x)
    decoded_imgs = predict_model(sess, decoder_model, encoded_imgs)
    display_images(valid_x, decoded_imgs,None)
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train Dropout Uncertainty Model',
            formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog))
    parser.add_argument('--forward', action='store_true', help='only forward model')
    parser.add_argument('--modelfn', type=str, default=None, help='model name')
    parser.add_argument('--output', type=str, default=None, help='model name')
    # learning hyper-parameters
    parser.add_argument('--init_lr', type=float, default=1e-2, help='initial learning rate.')
    parser.add_argument('--mom', type=float, default=0.9, help='SGD Momentum')
    parser.add_argument('--l2reg', type=float, default=5e-6, help='L2 reg lambda')
    parser.add_argument('--gamma', type=float, default=0.0001, help='L2 reg lambda')
    parser.add_argument('--power', type=float, default=0.25, help='L2 reg lambda')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')

    args = parser.parse_args()

    #USING RESMAN
    if args.output is None:
        args.output = os.environ.get('GIT_RESULTS_MANAGER_DIR',None)

    params = Empty()
    params.init_lr = args.init_lr
    params.num_epochs = args.num_epochs
    params.batch_size = args.batch_size
    params.l2reg = args.l2reg
    params.display_interval = 1
    params.modelfn = args.modelfn
    params.gamma = args.gamma
    params.power = args.power

    train_data, test_data = load_data()
    
    train_model(train_data, test_data,params)
