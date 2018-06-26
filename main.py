"""
Implementation of Conditionally-Shifted Neurons in Tensorflow
"""

import tensorflow as tf
import numpy as np

# ResBlock
# filters 64, 96, 128, 256
# dropout rate 0.2, 0.5, 0.5
# optimizer SGD with momentum lr=0.01, m=0.9

class ResBlock(object):

	def __init__(self, inputs, n_filters, name, is_training):
		super(ResBlock, self).__init__()
		self.inputs = inputs
		self.name = name
		with tf.scope(self.name):
			self.build_model(n_filters, is_training)

	def build_model(self, n_filters, is_training):
		conv_1 = tf.layers.conv2d(
			inputs=self.inputs,
			filters=n_filters,
			kernel_size=(3, 3),
			activation=tf.nn.relu,
		)
		bn_1 = tf.layers.batch_normalization(
			inputs=conv_1,
		)
		conv_2 = tf.layers.conv2d(
			inputs=bn_1,
			filters=n_filters,
			kernel_size=(3, 3),
			activation=tf.nn.relu,
		)
		bn_2 = tf.layers.batch_normalization(
			inputs=conv_2,
		)
		conv_3 = tf.layers.conv2d(
			inputs=bn_2,
			filters=n_filters,
			kernel_size=(3, 3),
			activation=tf.nn.relu,
		)
		bn_3 = tf.layers.batch_normalization(
			inputs=conv_3,
		)
		res_conv = tf.layers.conv2d(
			inputs=self.input,
			filters=n_filters,
			kernel_size=(1, 1),
		)
		max_pool = tf.layers.max_pooling2d(
			inputs=bn_3+res_conv,
			pool_size=(2, 2),
			strides=(1, 1),
		)
		self.outputs = tf.layers.dropout(
			inputs=max_pool,
			rate=0.9,
			training=is_training,
		)

class MiniImageNetModel(object):

	def __init__(self):
		super(MiniImageNetModel, self).__init__()
		self.build_model()

	def build_model(self):
		self.inputs = tf.placeholder(
			shape=(None, 84, 84, 3),
			dtype=tf.float32,
			name="inputs",
		)
		self.is_training = tf.placeholder(
			shape=(None),
			dtype=tf.bool,
			name="is_training",
		)
		output = ResBlock(self.inputs, 64, name="resblock_1", is_training=self.is_training)
		output = ResBlock(output, 96, name="resblock_2", is_training=self.is_training)
		output = ResBlock(output, 128, name="resblock_3", is_training=self.is_training)
		output = ResBlock(output, 256, name="resblock_4", is_training=self.is_training)
		output = tf.layers.conv2d(
			inputs=output,
			filters=1024,
			kernel_size=(1, 1),
			activation=tf.nn.relu,
		)
		output = tf.layers.average_pooling2d(
			inputs=output,
			pool_size=(6, 6),
			strides=(1, 1),
		)
		output = tf.layers.dropout(
			inputs=output,
			rate=0.9,
			training=is_training,
		)
		output = tf.layers.conv2d(
			inputs=output,
			filters=384,
			kernel_size=(1, 1),
			activation=tf.nn.relu,
		)
