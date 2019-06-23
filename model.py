import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random

from layers import *

input_img_fea_dim = 4096
input_txt_fea_dim = 300
cate_fea_size = 300
n_hidden_1 = 4096
n_hidden_2 = 4096


##############################################  DADN   ##############################################
def SeGAN_img(inputdisc,keep_prob,train,name): 

    with tf.variable_scope(name):
        layer_1 = fully_connected(inputdisc, n_hidden_1, 'fc1')
        layer_1 = tf.nn.relu(layer_1)  
        layer_1 = tf.nn.dropout(layer_1, keep_prob)
        layer_2 = fully_connected(layer_1, n_hidden_2, 'fc2')
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        out_layer = fully_connected(layer_2, cate_fea_size, 'fc3')
        return out_layer

def SeGAN_txt(inputdisc,keep_prob,train,name):

    with tf.variable_scope(name):

        layer_1 = fully_connected(inputdisc, n_hidden_1, 'fc1')
        layer_1 = tf.nn.relu(layer_1)  
        layer_1 = tf.nn.dropout(layer_1, keep_prob)
        layer_2 = fully_connected(layer_1, n_hidden_2, 'fc2')
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        out_layer = fully_connected(layer_2, cate_fea_size, 'fc3')
        return out_layer

def SeGAN_discriminator_img_MMD(inputdisc,oriFea,keep_prob,train,name):

    with tf.variable_scope(name):

        layer_0 = tf.concat([inputdisc, oriFea],1)
        layer_1 = fully_connected(layer_0, 4096, 'fc1')
        layer_1 = tf.nn.relu(layer_1)  
        layer_1 = tf.nn.dropout(layer_1, keep_prob)
        layer_2 = fully_connected(layer_1, 2048, 'fc2')
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        out_layer = fully_connected(layer_2, 1, 'fc3')
        return layer_0,layer_1,layer_2,out_layer



def SeGAN_discriminator_txt_MMD(inputdisc,oriFea,keep_prob,train,name):

    with tf.variable_scope(name):

        layer_0 = tf.concat([inputdisc, oriFea],1)
        layer_1 = fully_connected(layer_0, 4096, 'fc1')
        layer_1 = tf.nn.relu(layer_1)  
        layer_1 = tf.nn.dropout(layer_1, keep_prob)
        layer_2 = fully_connected(layer_1, 2048, 'fc2')
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        out_layer = fully_connected(layer_2, 1, 'fc3')
        return layer_0,layer_1,layer_2,out_layer

def ReGAN_img(inputdisc,keep_prob,train,name):

    with tf.variable_scope(name):

        layer_1 = fully_connected(inputdisc, 4096, 'fc1')
        layer_1 = tf.nn.relu(layer_1)  
        layer_1 = tf.nn.dropout(layer_1, keep_prob)
        layer_2 = fully_connected(layer_1, 4096, 'fc2')
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        out_layer = fully_connected(layer_2, input_img_fea_dim, 'fc3')
        return out_layer
     
def ReGAN_txt(inputdisc,keep_prob,train,name):

    with tf.variable_scope(name):

        layer_1 = fully_connected(inputdisc, 4096, 'fc1')
        layer_1 = tf.nn.relu(layer_1)  
        layer_1 = tf.nn.dropout(layer_1, keep_prob)
        layer_2 = fully_connected(layer_1, 4096, 'fc2')
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        out_layer = fully_connected(layer_2, input_txt_fea_dim, 'fc3')
        return out_layer


def ReGAN_discriminator_img_MMD(inputdisc,keep_prob,train,name):

    with tf.variable_scope(name):

        layer_1 = fully_connected(inputdisc, 4096, 'fc1')
        layer_1 = tf.nn.relu(layer_1)  
        layer_1 = tf.nn.dropout(layer_1, keep_prob)
        layer_2 = fully_connected(layer_1, 2048, 'fc2')
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        out_layer = fully_connected(layer_2, 1, 'fc3')
        return inputdisc,layer_1,layer_2,out_layer


def ReGAN_discriminator_txt_MMD(inputdisc,keep_prob,train,name):

    with tf.variable_scope(name):

        layer_1 = fully_connected(inputdisc, 4096, 'fc1')
        layer_1 = tf.nn.relu(layer_1)  
        layer_1 = tf.nn.dropout(layer_1, keep_prob)
        layer_2 = fully_connected(layer_1, 2048, 'fc2')
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        out_layer = fully_connected(layer_2, 1, 'fc3')
        return inputdisc,layer_1,layer_2,out_layer


