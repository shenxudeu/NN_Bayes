'''
Various Neural Network based Encoding Models
    - AutoEncoder
    - Variantional AutoEncoder
Reference: https://blog.keras.io/building-autoencoders-in-keras.html

http://blog.fastforwardlabs.com/post/148842796218/introducing-variational-autoencoders-in-prose-and
http://blog.fastforwardlabs.com/post/149329060653/under-the-hood-of-the-variational-autoencoder-in
https://github.com/fastforwardlabs/vae-tf
https://arxiv.org/pdf/1312.6114v10.pdf

From NN and Bayes views to see VAE
https://jaan.io/unreasonable-confusion/

'''


import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Model
from keras.engine import Layer
from keras.layers import Dense, Flatten, Input, Activation, Reshape, Dropout , Lambda
from keras.initializations import get_fans
from keras.regularizers import l2
from keras import regularizers
import keras.backend as K
from IPython import embed

def load_model(sess, saver, modelfn='save.ckpt'):
    if not os.path.isfile(modelfn):
        print '--- %s not found. Restart a new training process.' % modelfn
        return False
    saver.restore(sess, modelfn)
    print '---------------model restored'
    return True

def save_model(sess, saver, modelfn='save.ckpt'):
    save_path = saver.save(sess, modelfn)
    print 'Model saved in file: ', save_path

def make_loss_construction(model, in_x):
    model.construction_loss = tf.reduce_mean(K.binary_crossentropy(model.output, in_x))
    return model

def build_autoencoder(input_shape, l2reg = 0.):
    # this is the size of encoded representation
    encoding_dim_l1 = 128
    encoding_dim_l2 = 64
    encoding_dim = 32 

    (N, C) = input_shape

    in_x = Input(shape=(C,))
    
    # Encoding
    encoded = Dense(encoding_dim_l1, activation='relu', W_regularizer=l2(l2reg))(in_x)
    encoded = Dense(encoding_dim_l2, activation='relu', W_regularizer=l2(l2reg))(encoded)
    encoded = Dense(encoding_dim, activation='relu', W_regularizer=l2(l2reg))(encoded)

    # Decoding
    decoded = Dense(encoding_dim_l2, activation='relu', W_regularizer=l2(l2reg))(encoded)
    decoded = Dense(encoding_dim_l1, activation='relu', W_regularizer=l2(l2reg))(decoded)
    decoded = Dense(C, activation='sigmoid')(decoded)

    # Make Keras Models
    # End-To-End Model
    model = Model(input=in_x, output=decoded)
    model = make_loss_construction(model, in_x)
    model.loss = model.construction_loss

    # Encoder Model
    encoder_model = Model(input=in_x, output=encoded)

    # Decoder Model
    encoded_input = Input(shape=(encoding_dim,))
    decoded_ = model.layers[-3](encoded_input)
    decoded_ = model.layers[-2](decoded_)
    decoded_ = model.layers[-1](decoded_)
    decoder_model = Model(input=encoded_input, output=decoded_)

    return model, encoder_model, decoder_model


def make_ave_loss(model,z_mean, z_log_var):
    construction_loss = K.binary_crossentropy(model.output, model.input)
    KL_loss = -0.5 * K.sum(1+ z_log_var -K.square(z_mean) - K.exp(z_log_var),axis=-1)
    model.loss = tf.reduce_mean(K.mean(construction_loss,axis=-1) + KL_loss)
    return model

def build_ave(input_shape, l2reg = 0., n_latent=2,n_dim=500, n_dim2=501):
    latent_dim = n_latent
    intermediate_dim = n_dim
    intermediate_dim2 = n_dim2

    (N, C) = input_shape

    in_x = Input(shape=(C,))

    # Encoding
    h = Dense(intermediate_dim, activation='relu', W_regularizer=l2(l2reg))(in_x)
    h = Dense(intermediate_dim2, activation='relu', W_regularizer=l2(l2reg))(h)
    # z ~ N(z_mean, np.exp(z_log_var))
    z_mean = Dense(latent_dim)(h) # latent distribution mean
    z_log_var = Dense(latent_dim)(h) # latent distribution log-variance

    # Generate Samples Keras Layer
    def gaussian_sampling(args):
        '''(Differentiably!) draw samle from Gaussian with mu and log variance'''
        z_mean, z_log_var = args
        #epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0., std=0.01)
        epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0., std=0.002)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    z = Lambda(gaussian_sampling, output_shape=(latent_dim,))([z_mean,z_log_var])

    # Decoding
    decoder_h = Dense(intermediate_dim2, activation='relu',W_regularizer=l2(l2reg))
    decoder_h2 = Dense(intermediate_dim, activation='relu',W_regularizer=l2(l2reg))
    decoder_mean = Dense(C, activation='sigmoid',W_regularizer=l2(l2reg))
    h_decoded = decoder_h(z)
    h_decoded = decoder_h2(h_decoded)
    x_decoded_mean = decoder_mean(h_decoded)

    # Make Keras Models
    # End-To-End Model
    model = Model(input=in_x, output=x_decoded_mean)
    model = make_ave_loss(model, z_mean, z_log_var)

    # Encoder Model, from inputs to latent space
    encoder_model = Model(input=in_x, output=z_mean)

    # Generator Model, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _h_decoded = decoder_h2(_h_decoded)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator_model = Model(input=decoder_input, output=_x_decoded_mean)

    return model, encoder_model, generator_model


