from __future__ import division
import tensorflow as tf
from ops import *
from utils import *

def conv(inputs, reuse=False, name="encoder"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        net = slim.conv2d(input, 64, kernel_size=4, stride=2)
        f1a = slim.batch_norm(net, activation_fn = tf.nn.relu)
        net = slim.conv2d(f1a, 128, kernel_size=4, stride=2)
        f2a = slim.batch_norm(net, activation_fn = tf.nn.relu)
        net = slim.conv2d(f2a, 256, kernel_size=4, stride=2)
        f3a = slim.batch_norm(net, activation_fn = tf.nn.relu)
        net = slim.conv2d(f3a, 512, kernel_size=4, stride=2)
        f4a = slim.batch_norm(net, activation_fn = tf.nn.relu)

        return [f1a, f2a, f3a, f4a]

def attribute_classifier(inputs, reuse=False, name="attentive_editor"):

    with tf.variable_scope(name):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        net = slim.conv2d(input, 64, kernel_size=4, stride=2)
        net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)
        net = slim.conv2d(net, 128, kernel_size=4, stride=2)
        net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)
        net = slim.conv2d(net, 256, kernel_size=4, stride=2)
        net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)
        net = slim.conv2d(net, 512, kernel_size=4, stride=2)
        net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)
        net = slim.conv2d(net, 1024, kernel_size=4, stride=2)
        net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1024, activation_fn=tf.nn.leaky_relu)
        net = slim.fully_connected(net, 13, activation_fn=tf.sigmoid)

        return net

def discriminator(inputs, reuse=False, name="attentive_editor"):

    with tf.variable_scope(name):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        net = slim.conv2d(input, 64, kernel_size=4, stride=2)
        net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)
        net = slim.conv2d(net, 128, kernel_size=4, stride=2)
        net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)
        net = slim.conv2d(net, 256, kernel_size=4, stride=2)
        net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)
        net = slim.conv2d(net, 512, kernel_size=4, stride=2)
        net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)
        net = slim.conv2d(net, 1024, kernel_size=4, stride=2)
        net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1024, activation_fn=tf.nn.leaky_relu)
        net = slim.fully_connected(net, 1)

        return net

def attentive_editor_ek(inputs, k, reuse=False, name="attentive_editor_ek"):

    with tf.variable_scope(name):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        net = slim.deconv2d(inputs, 64*2^(k-1), kernel_size=3, stride=1)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        
        return net

def attentive_editor_ez(inputs, reuse=False, name="attentive_editor_e0"):

    with tf.variable_scope(name):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        net = slim.deconv2d(inputs, 32, kernel_size=3, stride=1)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        net = slim.deconv2d(net, 3, 3, 2, activation_fn=tf.nn.tanh)

        return net


def attiribute_predictor(inputs, reuse=False, name="attiribute_predictor"):

    with tf.variable_scope(name):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        net = slim.conv2d(inputs, 16, 3, 1)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        net = slim.conv2d(net, 16, 3, 1)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        net = slim.pool(net,2,stride=2)

        net = slim.conv2d(net, 32, 3, 1)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        net = slim.conv2d(net, 32, 3, 1)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        net = slim.pool(net,2,stride=2)

        net = slim.conv2d(net, 64, 3, 1)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        net = slim.conv2d(net, 64, 3, 1)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        net = slim.pool(net,2,stride=2)

        net = slim.conv2d(net, 128, 3, 1)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        net = slim.conv2d(net, 128, 3, 1)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        net = slim.pool(net,2,stride=2)

        net = slim.flatten(net)
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
        net = slim.fully_connected(net, 40, activation_fn=tf.sigmoid)

        return net

def generator(inputs, eta, reuse=False, name="generator"):

    with tf.variable_scope("generator"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        net = slim.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, weights_initializer=tf.initializers.he_normal())
        net = slim.fully_connected(net, 784, activation_fn=None, weights_initializer=tf.initializers.he_normal())

        return net

def e_encoder(inputs, reuse=False, name="e_encoder"):

    with tf.variable_scope("e_encoder"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        net = slim.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, weights_initializer=tf.initializers.he_normal())
        net = slim.fully_connected(net, 50, activation_fn=None, weights_initializer=tf.initializers.he_normal())

        return net

def discriminator(inputs, reuse=False, name="discriminator"):

    with tf.variable_scope("discriminator"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        net = slim.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, weights_initializer=tf.initializers.he_normal())
        net = slim.fully_connected(net, 1, activation_fn=None, weights_initializer=tf.initializers.he_normal())

        return net



def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target): # mae??? not mse??
    return tf.reduce_mean(tf.abs(in_-target))

def mse_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
