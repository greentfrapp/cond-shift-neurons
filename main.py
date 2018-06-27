"""
Implementation of Conditionally-Shifted Neurons in Tensorflow
"""

import tensorflow as tf
import numpy as np
from absl import flags
from absl import app

FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")

# Training parameters
flags.DEFINE_string("savepath", "models/", "Path to save or load models")

# ResBlock
class ResBlock(object):

	def __init__(self, inputs, n_filters, name, is_training):
		super(ResBlock, self).__init__()
		self.inputs = inputs
		self.name = name
		with tf.variable_scope(self.name):
			self.build_model(n_filters, is_training)

	def build_model(self, n_filters, is_training):
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
		self.outputs = tf.layers.dropout(
			inputs=max_pool,
			rate=0.5,
			training=is_training,
		)

# MiniImageNetModel comprising several ResBlocks
class MiniImageNetModel(object):

	def __init__(self, name, k=5):
		super(MiniImageNetModel, self).__init__()
		self.name = name
		with tf.variable_scope(self.name):
			self.build_model(k)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self, k):
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
		resblock_1 = ResBlock(self.inputs, 64, name="resblock_1", is_training=self.is_training)
		output = resblock_1.outputs
		resblock_2 = ResBlock(output, 96, name="resblock_2", is_training=self.is_training)
		output = resblock_2.outputs
		resblock_3 = ResBlock(output, 128, name="resblock_3", is_training=self.is_training)
		output = resblock_3.outputs
		resblock_4 = ResBlock(output, 256, name="resblock_4", is_training=self.is_training)
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
		self.predictions = tf.argmax(self.logits, axis=1)
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
		self.optimize = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9).minimize(self.loss)

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


def main(unused_args):
	
	if FLAGS.train:
		from keras.datasets import mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		model = MiniImageNetModel("mnist", k=10)
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		n_steps = 1000
		batchsize = 64
		rand_x = np.random.RandomState(1)
		rand_y = np.random.RandomState(1)
		start = 0
		for i in np.arange(n_steps):
			end = int(start + batchsize)
			if end > len(x_train):
				end -= len(x_train)
				minibatch_x_1 = np.expand_dims(x_train[start:], axis=3) / 255.
				minibatch_y_1 = y_train[start:]
				rand_x.shuffle(x_train)
				rand_y.shuffle(y_train)
				minibatch_x_2 = np.expand_dims(x_train[:end], axis=3) / 255.
				minibatch_y_2 = y_train[:end]
				minibatch_x = np.concatenate([minibatch_x_1, minibatch_x_2], axis=0)
				minibatch_y = np.concatenate([minibatch_y_1, minibatch_y_2], axis=0)
			else:
				minibatch_x = np.expand_dims(x_train[start:end], axis=3) / 255.
				minibatch_y = y_train[start:end]
			start = end
			feed_dict = {
				model.inputs: minibatch_x,
				model.labels: minibatch_y,
				model.is_training: True,
			}
			loss, _ = sess.run([model.loss, model.optimize], feed_dict)
			if (i + 1) % 50 == 0:
				print("Step {} - Loss {:.3f}".format(i + 1, loss))
				model.save(sess, FLAGS.savepath, global_step=i+1)

	if FLAGS.test:
		from keras.datasets import mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		model = MiniImageNetModel("mnist", k=10)
		sess = tf.Session()
		model.load(sess, FLAGS.savepath)

		batchsize = 256
		start = 0
		predictions = None
		i = 0
		while True:
			end = int(start + batchsize)
			if start >= len(x_train):
				break
			minibatch_x = np.expand_dims(x_train[start:end], axis=3) / 255.
			minibatch_y = y_train[start:end]
			start = end
			feed_dict = {
				model.inputs: minibatch_x,
				model.is_training: False,
			}
			batch_predictions = sess.run(model.predictions, feed_dict)

			if predictions is None:
				predictions = batch_predictions
			else:
				predictions = np.concatenate([predictions, batch_predictions], axis=0)

			if (i + 1) % 50 == 0:
				print("Predicted {} samples...".format(end))

			i += 1
			
		print(np.sum(predictions == y_train) / len(y_train))





if __name__ == "__main__":
	app.run(main)
