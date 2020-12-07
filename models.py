from __future__ import print_function, division
import scipy

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import Reshape
import datetime
import sys
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
import numpy as np
import os
import random
from keras.layers import Conv2DTranspose, BatchNormalization
import tensorflow as tf

from keras.utils import to_categorical


def get_dim_conv(dim, f, p, s):
    return int((dim + 2 * p - f) / 2 + 1)


def build_generator_enc_dec(img_shape, gf, num_classes, channels, num_layers=4, f_size=4, tranform_layer=False):
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=f_size):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='valid')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        return d

    def __deconv2d(layer_input, skip_input, filters, f_size=f_size, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='valid', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = Concatenate()([u, skip_input])
        return u

    def deconv2d(layer_input, skip_input, filters, f_size=f_size, dropout_rate=0, output_padding=None):
        """Layers used during upsampling"""
        u = Conv2DTranspose(filters=filters, kernel_size=f_size,
                            strides=2, activation='relu', output_padding=output_padding)(layer_input)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    img = Input(shape=img_shape)

    # Downsampling
    d = img
    zs = []
    dims = []
    _dim = img_shape[0]
    for i in range(num_layers):
        d = conv2d(d, gf * 2 ** i)
        zs.append(d)
        _dim = get_dim_conv(_dim, f_size, 0, 2)
        dims.append((_dim, gf * 2 ** i))
        print("D:", _dim, gf * 2 ** i)
    G_enc = Model(img, zs)

    _zs = []
    d_, c_ = dims.pop()
    print(0, d_, c_)
    i_ = Input(shape=(d_, d_, c_))
    _zs.append(i_)
    label = Input(shape=(num_classes,), dtype='float32')
    label_r = Reshape((1, 1, num_classes))(label)

    u = concatenate([i_, label_r], axis=-1)

    ## transf
    if tranform_layer:
        tr = Flatten()(u)
        tr = Dense(c_ + num_classes)(tr)
        tr = LeakyReLU(alpha=0.2)(tr)
        u = Reshape((1, 1, c_ + num_classes))(tr)
    ##

    u = Conv2D(c_, kernel_size=1, strides=1, padding='valid')(u)  ## 1x1 conv

    # Upsampling
    for i in range(num_layers - 1):
        _ch = gf * 2 ** ((num_layers - 2) - i)
        d_, c_ = dims.pop()
        print(i, d_, c_)
        i_ = Input(shape=(d_, d_, c_))
        _zs.append(i_)
        if i == 2:
            u = deconv2d(u, i_, _ch, output_padding=1)
        else:
            u = deconv2d(u, i_, _ch)

    u = Conv2DTranspose(filters=channels, kernel_size=f_size,
                        strides=2, activation='tanh', output_padding=None)(u)

    _zs.reverse()
    _zs.append(label)
    G_dec = Model(_zs, u)

    return G_enc, G_dec


def build_discriminator(img_shape, df, num_classes, num_layers=4, act_multi_label='softmax'):
    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='valid')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        return d

    img = Input(shape=img_shape)

    d = img
    for i in range(num_layers):
        _norm = False if i == 0 else True
        d = d_layer(d, df * 2 ** i, normalization=_norm)

    flat_repr = Flatten()(d)

    gan_logit = Dense(df * 2 ** (num_layers - 1))(flat_repr)
    gan_logit = LeakyReLU(alpha=0.2)(gan_logit)
    gan_prob = Dense(1, activation='sigmoid')(gan_logit)

    class_logit = Dense(df * 2 ** (num_layers - 1))(flat_repr)
    class_logit = LeakyReLU(alpha=0.2)(class_logit)
    class_prob = Dense(num_classes, activation=act_multi_label)(class_logit)

    return Model(img, [gan_prob, class_prob])
