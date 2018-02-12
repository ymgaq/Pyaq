# -*- coding: utf-8 -*-

from board import *
import tensorflow as tf

FILTER_CNT = 96
BLOCK_CNT = 6

w_wdt = 0.007
b_wdt = 0.015


class DualNetwork(object):

    def get_variable(self, shape_, width_=0.007, name_="weight"):
        var = tf.get_variable(name_, shape=shape_,
                              initializer=tf.random_normal_initializer(
                                  mean=0, stddev=width_))

        if not tf.get_variable_scope()._reuse:
            tf.add_to_collection("vars_train", var)

        return var

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1],
                            padding='SAME', name="conv2d")

    def res_block(self, x, input_size, middle_size, output_size,
                  dr_block=1.0, scope_name="res"):

        with tf.variable_scope(scope_name + "_0"):
            w0 = self.get_variable([3, 3, input_size, middle_size],
                                   w_wdt, name_="weight")
            b0 = self.get_variable([middle_size], b_wdt, name_="bias")
            conv0 = tf.nn.relu(self.conv2d(x, w0) + b0)
        with tf.variable_scope(scope_name + "_1"):
            w1 = self.get_variable([3, 3, middle_size, output_size],
                                   w_wdt, name_="weight")
            b1 = self.get_variable([output_size], b_wdt, name_="bias")
            conv1 = tf.nn.dropout(self.conv2d(conv0, w1) + b1, dr_block)

        if input_size == output_size:
            x_add = x
        elif input_size < output_size:
            x_add = tf.pad(x, [[0, 0], [0, 0], [0, 0],
                               [0, output_size - input_size]])
        else:
            x_add = tf.slice(x, [0, 0, 0, 0],
                             [-1, BSIZE, BSIZE, output_size])

        return tf.nn.relu(tf.add(conv1, x_add))

    def model(self, x, temp=1.0, dr=1.0):
        hi = []
        prev_h = tf.reshape(x, [-1, BSIZE, BSIZE, FEATURE_CNT])

        # residual blocks with N layers
        for i in range(BLOCK_CNT):
            input_size = FEATURE_CNT if i == 0 else FILTER_CNT
            dr_block = 1 - (1 - dr) / BLOCK_CNT * i

            hi.append(self.res_block(prev_h, input_size, FILTER_CNT, FILTER_CNT,
                                     dr_block=dr_block, scope_name="res%d" % i))
            prev_h = hi[i]

        # policy connection
        with tf.variable_scope('pfc'):
            # 1st layer
            # [-1, BSIZE, BSIZE, FILTER_CNT] => [-1, BSIZE**2 * 2]
            w_pfc0 = self.get_variable([1, 1, FILTER_CNT, 2],
                                       w_wdt, name_="weight0")
            b_pfc0 = self.get_variable([BSIZE, BSIZE, 2], b_wdt, name_="bias0")
            conv_pfc0 = tf.reshape(self.conv2d(hi[BLOCK_CNT - 1], w_pfc0)
                                   + b_pfc0, [-1, BVCNT * 2])

            # 2nd layer
            # [-1, BSIZE**2 * 2] => [-1, BSIZE**2 + 1]
            w_pfc1 = self.get_variable([BVCNT * 2, BVCNT + 1],
                                       w_wdt, name_="weight1")
            b_pfc1 = self.get_variable([BVCNT + 1], b_wdt, name_="bias1")
            conv_pfc1 = tf.matmul(conv_pfc0, w_pfc1) + b_pfc1

            # divided by softmax temp and apply softmax
            policy = tf.nn.softmax(tf.div(conv_pfc1, temp), name="policy")

        # value connection
        with tf.variable_scope('vfc'):
            # 1st layer
            # [-1, BSIZE, BSIZE, FILTER_CNT] => [-1, BSIZE**2]
            w_vfc0 = self.get_variable([1, 1, FILTER_CNT, 1],
                                       w_wdt, name_="weight0")
            b_vfc0 = self.get_variable([BSIZE, BSIZE, 1], b_wdt, name_="bias0")
            conv_vfc0 = tf.reshape(self.conv2d(hi[BLOCK_CNT - 1], w_vfc0)
                                   + b_vfc0, [-1, BVCNT])

            # 2nd layer
            # [-1, BSIZE**2] => [-1, 256]
            w_vfc1 = self.get_variable([BVCNT, 256], w_wdt, name_="weight1")
            b_vfc1 = self.get_variable([256], b_wdt, name_="bias1")
            conv_vfc1 = tf.matmul(conv_vfc0, w_vfc1) + b_vfc1
            relu_vfc1 = tf.nn.relu(conv_vfc1)

            # 3rd layer
            # [-1, 256] => [-1, 1]
            w_vfc2 = self.get_variable([256, 1], w_wdt, name_="weight2")
            b_vfc2 = self.get_variable([1], b_wdt, name_="bias2")
            conv_vfc2 = tf.matmul(relu_vfc1, w_vfc2) + b_vfc2

            # apply tanh
            value = tf.nn.tanh(tf.reshape(conv_vfc2, [-1]), name="value")

        return policy, value

    def create_sess(self, ckpt_path=""):
        with tf.get_default_graph().as_default():

            sess_ = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False))
            vars_train = tf.get_collection("vars_train")
            v_to_init = list(set(tf.global_variables()) - set(vars_train))

            saver = tf.train.Saver(vars_train, write_version=1)
            if ckpt_path != "":
                saver.restore(sess_, ckpt_path)
                sess_.run(tf.variables_initializer(v_to_init))
            else:
                sess_.run(tf.global_variables_initializer())

        return sess_

    def save_vars(self, sess_, ckpt_path="model.ckpt"):
        with tf.get_default_graph().as_default():

            vars_train = tf.get_collection("vars_train")
            saver = tf.train.Saver(vars_train, write_version=1)
            saver.save(sess_, ckpt_path)
