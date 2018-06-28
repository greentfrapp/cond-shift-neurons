"""
Implementation of Conditionally-Shifted Neurons in Tensorflow

Outline:
There will be two MiniImageNetModels, with identical weights.
There is also gradient embedder G.
During training, the training samples are fed into net A.
We then obtain gradients from A and pass to G to update the memory matrix.
Net B then take a weighted sum of the memory matrix to do prediction.

So we can backpropagate through both B and G during meta-training.

During meta-test, we only need one copy of the MiniImageNetModel.

TODO:
- Add copy op to MiniImageNetModel
- Think of how to calculate csn values

"""

import tensorflow as tf
import numpy as np

def update_target_graph(from_scope,to_scope):
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
	op_holder = []
	for from_var,to_var in zip(from_vars,to_vars):
		op_holder.append(to_var.assign(from_var))
	return op_holder

# ResBlock
class ResBlock(object):

	def __init__(self, inputs, n_filters, name, is_training, csn=None):
		super(ResBlock, self).__init__()
		self.inputs = inputs
		self.name = name
		with tf.variable_scope(self.name):
			self.build_model(n_filters, is_training, csn)

	def build_model(self, n_filters, is_training=False, csn=None):
		conv_1 = tf.layers.conv2d(
			inputs=self.inputs,
			filters=n_filters,
			kernel_size=(3, 3),
			padding="same",
			activation=tf.nn.relu,
		)
		bn_1 = tf.contrib.layers.batch_norm(
			inputs=conv_1,
		)
		conv_2 = tf.layers.conv2d(
			inputs=bn_1,
			filters=n_filters,
			kernel_size=(3, 3),
			padding="same",
			activation=tf.nn.relu,
		)
		bn_2 = tf.contrib.layers.batch_norm(
			inputs=conv_2,
		)
		conv_3 = tf.layers.conv2d(
			inputs=bn_2,
			filters=n_filters,
			kernel_size=(3, 3),
			padding="same",
			activation=tf.nn.relu,
		)
		bn_3 = tf.contrib.layers.batch_norm(
			inputs=conv_3,
		)
		res_conv = tf.layers.conv2d(
			inputs=self.inputs,
			filters=n_filters,
			kernel_size=(1, 1),
			padding="same",
			activation=None,
		)
		max_pool = tf.layers.max_pooling2d(
			inputs=bn_3+res_conv,
			pool_size=(2, 2),
			strides=(1, 1),
		)
		# seems like the gradient should be added prior to the relu
		if csn is not None:
			max_pool -= 1e-4 * csn
		output = tf.nn.relu(max_pool)
		self.outputs = tf.layers.dropout(
			inputs=max_pool,
			rate=0.5,
			training=is_training,
		)
		self.gradients = tf.gradients(self.outputs, max_pool)

# MiniImageNetModel comprising several ResBlocks
class MiniImageNetModel(object):

	def __init__(self, name, k=5, csn=None):
		super(MiniImageNetModel, self).__init__()
		self.name = name
		with tf.variable_scope(self.name):
			self.build_model(k, csn)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self, k, csn=None):
		self.inputs = tf.placeholder(
			shape=(None, 28, 28, 1),
			dtype=tf.float32,
			name="inputs",
		)
		self.labels = tf.placeholder(
			shape=(None),
			dtype=tf.int32,
			name="labels",
		)
		self.is_training = tf.placeholder(
			shape=(None),
			dtype=tf.bool,
			name="is_training",
		)
		if csn is None:
			csn = {
				"resblock_3": None,
				"resblock_4": None,
				"logits": None,
			}
		resblock_1 = ResBlock(self.inputs, 64, name="resblock_1", is_training=self.is_training)
		output = resblock_1.outputs
		resblock_2 = ResBlock(output, 96, name="resblock_2", is_training=self.is_training)
		output = resblock_2.outputs
		resblock_3 = ResBlock(output, 128, name="resblock_3", is_training=self.is_training, csn=csn["resblock_3"])
		output = resblock_3.outputs
		resblock_4 = ResBlock(output, 256, name="resblock_4", is_training=self.is_training, csn=csn["resblock_4"])
		output = resblock_4.outputs
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
			rate=0.5,
			training=self.is_training,
		)
		output = tf.layers.conv2d(
			inputs=output,
			filters=384,
			kernel_size=(1, 1),
			activation=None,
		)
		output = tf.reshape(output, [-1, 19 * 19 * 384])
		output = tf.layers.dense(
			inputs=output,
			units=k,
			activation=None,
		)
		
		self.logits = output
		if csn["logits"] is not None:
			self.logits -= 1e-4 * csn["logits"]
		self.predictions = tf.argmax(self.logits, axis=1)
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

		if csn["logits"] is None:
			self.optimize = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9).minimize(self.loss)

			self.csn_gradients = {
				"resblock_3": resblock_3.gradients[0],
				"resblock_4": resblock_4.gradients[0],
				"logits": tf.gradients(self.loss, self.logits)[0],
			}

	def save(self, sess, savepath, global_step=None, prefix="ckpt", verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		self.saver.save(sess, savepath + prefix, global_step=global_step)
		if verbose:
			print("Model saved to {}.".format(savepath + prefix + '-' + str(global_step)))

	def load(self, sess, savepath, verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		ckpt = tf.train.latest_checkpoint(savepath)
		self.saver.restore(sess, ckpt)
		if verbose:
			print("Model loaded from {}.".format(ckpt))
