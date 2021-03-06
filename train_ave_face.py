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

from model_builder import build_autoencoder, build_ave, save_model, load_model
#from keras.datasets import mnist
from face_data.face_data import load_data

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def pre_process(x):
    x = 0.2989*x[:,:,:,0] + 0.5870*x[:,:,:,1] + 0.1140 * x[:,:,:,2]
    x = x[:,::4,::4]
    return x

class Empty(object):
    def __init__(self,name='duck'):
        self.name = name


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

def display_latent(latent_var,y):
    plt.figure(figsize=(6,6))
    plt.scatter(latent_var[:,0],latent_var[:,1],c = y)
    plt.colorbar()
    plt.show()


def predict_model(sess, model, x):
    feed_dict = {
            model.input: x,
            K.learning_phase(): 0}
    pred_vals = sess.run(model.output, feed_dict=feed_dict)   
    return pred_vals

def generate_images(train_data, valid_data, p):
    (train_x, train_y), (valid_x, valid_y) = train_data, test_data
    train_x = pre_process(train_x)
    valid_x = pre_process(valid_x)
    
    train_x = train_x.reshape(len(train_x),-1) / 255.
    valid_x = valid_x.reshape(len(valid_x),-1) / 255.

    in_shape = (None, train_x.shape[1])
    model, encoder_model, generator_model = build_ave(in_shape, p.l2reg,p.n_latent,p.n_dim,p.n_dim2)

    learning_rate = tf.placeholder(tf.float32, shape=[])

    tfepoch = tf.Variable(0)
    saver = tf.train.Saver()

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    load_model(sess, saver, modelfn=p.modelfn)

    #encoded_imgs = predict_model(sess, encoder_model, valid_x)
    encoded_imgs = predict_model(sess, encoder_model, train_x)
    decoded_imgs = predict_model(sess, generator_model, encoded_imgs)
    embed()
    HERE

    # trained face latent is ranged at (-0.05, 0.05)
    current_var = np.ones(p.n_latent) * (-0.05)
    step = 0.01
    idx = 0
    for latent_id in range(0,p.n_latent,4):
        latent_ids = [latent_id, latent_id+1]
        for seed in np.arange(-0.05,0.05,step):
            current_var[latent_ids] = [seed,seed]
            latent_vars = np.expand_dims(current_var + np.random.normal(0,0.01,p.n_latent),0)
            #latent_vars = np.expand_dims(np.ones(12)*i + np.random.normal(0,0.001,12),0)
            #latent_vars = np.expand_dims(np.random.normal(0,0.15,12),0)
            digit = predict_model(sess, generator_model, latent_vars)[0].reshape(63,63)
            plt.imshow(digit,cmap='Greys_r')
            plt.savefig('figures/faces/img_%d.png'%idx, dpi=100)
            idx += 1
        
    sess.close()
    print '%d images have been generated'%(idx + 1)

def train_model(train_data, valid_data, p):
    (train_x, train_y), (valid_x, valid_y) = train_data, test_data
    train_x = pre_process(train_x)
    valid_x = pre_process(valid_x)
    
    train_x = train_x.reshape(len(train_x),-1) / 255.
    valid_x = valid_x.reshape(len(valid_x),-1) / 255.

    lr = p.init_lr

    in_shape = (None, train_x.shape[1])
    model, encoder_model, generator_model = build_ave(in_shape, p.l2reg,p.n_latent,p.n_dim,p.n_dim2)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)
    #train_step = tf.train.RMSPropOptimizer(learning_rate=0.001,decay=0.9,epsilon=1e-8).minimize(model.loss)

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
        save_model(sess, saver,p.modelfn)
    
    # generate decoded images to show
    encoded_imgs = predict_model(sess, encoder_model, valid_x)
    decoded_imgs = predict_model(sess, generator_model, encoded_imgs)
    #display_latent(encoded_imgs, valid_y)
    #display_images(valid_x, decoded_imgs,None)

    ## display a 2D manifold of the digits
    #n = 30 # figure with 15 x 15 digits
    #digit_size = 28 # determined by the trained img size
    #figure = np.zeros((digit_size * n, digit_size * n))
    ## we will sample n points within [-15, 15] standard deviations
    #grid_x = np.linspace(-n, n, n)
    #grid_y = np.linspace(-n, n, n)
    #for i, yi in enumerate(grid_x):
    #    for j, xi in enumerate(grid_y):
    #        z_sample = np.array([xi, yi]) * 0.01
    #        x_decoded = predict_model(sess, generator_model, np.expand_dims(z_sample,0))
    #        digit = x_decoded[0].reshape(digit_size,digit_size)
    #        figure[i * digit_size: (i + 1) * digit_size,
    #               j * digit_size: (j + 1) * digit_size] = digit
    #plt.figure(figsize=(10,10))
    #plt.imshow(figure)
    #plt.show()

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train Dropout Uncertainty Model',
            formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog))
    parser.add_argument('--forward', action='store_true', help='only forward model')
    parser.add_argument('--modelfn', type=str, default=None, help='model name')
    parser.add_argument('--output', type=str, default=None, help='model name')
    # learning hyper-parameters
    parser.add_argument('--init_lr', type=float, default=1e-4, help='initial learning rate.')
    parser.add_argument('--mom', type=float, default=0.9, help='SGD Momentum')
    parser.add_argument('--l2reg', type=float, default=5e-6, help='L2 reg lambda')
    parser.add_argument('--gamma', type=float, default=0.0001, help='L2 reg lambda')
    parser.add_argument('--power', type=float, default=0.25, help='L2 reg lambda')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--n_latent', type=int, default=200, help='dim of latent space')
    
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
    params.n_latent = args.n_latent
    params.n_dim = 2000
    params.n_dim2 = 2000

    train_data, test_data,tags = load_data()
    embed()
    if args.forward:
        generate_images(train_data, test_data, params)
    else:
        train_model(train_data, test_data,params)


