# -*- coding: utf-8 -*-
"""
Created on April 8 2020
@author: Junchao Zhang
Hyperspectral Image Recovery
"""

import tensorflow as tf
import numpy as np

def Dense_block(x):
    for i in range(7):
        shape = x.get_shape().as_list()
        w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9.0 / shape[-1]))
        t = tf.layers.conv2d(x,16,3,(1,1),padding='SAME',kernel_initializer=w_init)
        t = tf.nn.relu(t)
        x = tf.concat([x,t],3)
    return x



def conv_layer(x,filternum,filtersize=3,isactiv=True,padding='SAME',use_bias = True):
    shape = x.get_shape().as_list()
    w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / filtersize/filtersize/ shape[-1]))
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    t = tf.layers.conv2d(x, filternum, filtersize, (1, 1), padding=padding, kernel_initializer=w_init,kernel_regularizer=regularizer,use_bias=use_bias)
    if isactiv:
        t = tf.nn.relu(t)
    return t

def forward(x):
    output = conv_layer(x, 16,3)
    output = Dense_block(output)

    output = conv_layer(output, 128, 3)
    output = conv_layer(output, 64, 3)
    output = conv_layer(output,31, 3)
    return output

