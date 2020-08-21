#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#	以下代码参考《Tensorflow实战》，与博客对应，可直接运行，有任何问题欢迎私我。
#	作者：柳玉豹 liuyubao96@gmail.com QQ:523331232

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y + 1e-10), reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

tf.global_variables_initializer().run()
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
