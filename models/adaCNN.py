"""
models - adaCNN
"""

import tensorflow as tf
import numpy as np

from models.utils import BaseModel, MemoryValueModel


# CNN used in adaCNN
class adaCNNNet(object):

	def __init__(self, name, inputs, layers, output_dim, parent, is_training, csn):
		super(adaCNNNet, self).__init__()
		self.name = name
		self.inputs = inputs
		self.is_training = is_training
		self.csn = csn
		self.gradients = dict()
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(layers, output_dim)

	def build_model(self, layers, output_dim):
		running_output = self.inputs
		for i in np.arange(layers):
			conv = tf.layers.conv2d(
				inputs=running_output,
				filters=32,
				kernel_size=(3, 3),
				padding="same",
				activation=None,
				name="conv_{}".format(i),
				reuse=tf.AUTO_REUSE,
			)
			if self.csn is not None and self.csn["conv_{}".format(i)] is not None:
				conv += self.csn["conv_{}".format(i)]
			relu = tf.nn.relu(conv)
			self.gradients["conv_{}".format(i)] = tf.gradients(relu, conv)
			maxpool = tf.layers.max_pooling2d(
				inputs=relu,
				pool_size=(2, 2),
				strides=(1, 1),
			)
			running_output = maxpool

		self.output = tf.layers.dense(
			inputs=tf.reshape(running_output, [-1, (28 - layers) * (28 - layers) * 32]),
			units=output_dim,
			activation=None,
			name="logits",
			reuse=tf.AUTO_REUSE,
		)
		self.logits = self.output
		if self.csn is not None:
			self.logits += self.csn["logits"]

class adaCNNModel(BaseModel):

	def __init__(self, name, num_classes=5, input_tensors=None, lr=1e-4, logdir=None, prefix='', is_training=None, num_test_classes=None):
		super(adaCNNModel, self).__init__()
		self.name = name
		# Use a mask to test on tasks with fewer classes than training tasks
		self.num_test_classes = num_test_classes
		if self.num_test_classes is not None:
			self.logit_mask = np.zeros([1, num_classes])
			for i in np.arange(num_test_classes):
				self.logit_mask[0, i] = 1
		else:
			self.logit_mask = np.ones([1, num_classes])
			self.num_test_classes = num_classes
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(num_classes, input_tensors, lr, is_training)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)
		if logdir is not None:
			self.writer = tf.summary.FileWriter(logdir + prefix)
			self.summary = tf.summary.merge([
				tf.summary.scalar("loss", self.test_loss, family=prefix),
				tf.summary.scalar("accuracy", self.test_accuracy, family=prefix),
			])

	def build_model(self, num_classes, input_tensors=None, lr=1e-4, is_training=None):

		if input_tensors is None:
			self.train_inputs = tf.placeholder(
				shape=(None, 28, 28, 1),
				dtype=tf.float32,
				name="train_inputs",
			)
			self.train_labels = tf.placeholder(
				shape=(None, num_classes),
				dtype=tf.float32,
				name="train_labels",
			)
			self.test_inputs = tf.placeholder(
				shape=(None, 28, 28, 1),
				dtype=tf.float32,
				name="test_inputs"
			)
			self.test_labels = tf.placeholder(
				shape=(None, num_classes),
				dtype=tf.float32,
				name="test_labels",
			)

		else:
			self.train_inputs = tf.reshape(input_tensors['train_inputs'], [-1, 28, 28, 1])
			self.test_inputs = tf.reshape(input_tensors['test_inputs'], [-1, 28, 28, 1])
			if tf.shape(input_tensors['train_labels'])[-1] != self.num_test_classes:
				self.train_labels = tf.reshape(tf.one_hot(tf.argmax(input_tensors['train_labels'], axis=2), depth=num_classes), [-1, num_classes])
				self.test_labels = tf.reshape(tf.one_hot(tf.argmax(input_tensors['test_labels'], axis=2), depth=num_classes), [-1, num_classes])
			else:
				self.train_labels = tf.reshape(input_tensors['train_labels'], [-1, num_classes])
				self.test_labels = tf.reshape(input_tensors['test_labels'], [-1, num_classes])
		if is_training is None:
			self.is_training = tf.placeholder(
				shape=(None),
				dtype=tf.bool,
				name="is_training",
			)
		else:
			self.is_training = is_training

		batch_size = tf.shape(input_tensors['train_inputs'])[0]

		self.inputs = tf.concat([self.train_inputs, self.test_inputs], axis=0)
		self.labels = tf.concat([self.train_labels, self.test_labels], axis=0)

		# CNN

		self.cnn_train = adaCNNNet(
			name="cnn", 
			inputs=self.train_inputs,
			layers=4,
			output_dim=num_classes,
			parent=self,
			is_training=self.is_training, 
			csn=None
		)
		
		# Need to calculate training loss per task
		self.train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(self.train_labels, [batch_size, -1, num_classes]), logits=tf.reshape(self.cnn_train.logits, [batch_size, -1, num_classes])), axis=1)

		# Preshift accuracy for logging
		self.train_predictions = tf.argmax(self.cnn_train.logits, axis=1)
		self.train_accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(self.train_labels, axis=1), predictions=self.train_predictions)
		
		# CSN Memory Matrix

		# - Keys

		self.memory_key_model = adaCNNNet(
			name="key_model",
			inputs=self.inputs,
			layers=2,
			output_dim=32,
			parent=self,
			is_training=self.is_training,
			csn=None,
		)
		keys = tf.split(
			self.memory_key_model.output,
			[tf.shape(self.train_inputs)[0], tf.shape(self.test_inputs)[0]],
			axis=0,
		)
		self.train_keys = train_keys = tf.reshape(keys[0], [batch_size, -1, 32])
		self.test_keys = test_keys = tf.reshape(keys[1], [batch_size, -1, 32])

		# - Values

		csn_gradients = {
			"conv_1": tf.reshape(self.cnn_train.gradients["conv_1"][0], [-1, 27 * 27 * 32, 1]) * tf.expand_dims(tf.gradients(self.train_loss, self.cnn_train.logits)[0], axis=1),
			"conv_2": tf.reshape(self.cnn_train.gradients["conv_2"][0], [-1, 26 * 26 * 32, 1]) * tf.expand_dims(tf.gradients(self.train_loss, self.cnn_train.logits)[0], axis=1),
			"conv_3": tf.reshape(self.cnn_train.gradients["conv_3"][0], [-1, 25 * 25 * 32, 1]) * tf.expand_dims(tf.gradients(self.train_loss, self.cnn_train.logits)[0], axis=1),
			# "conv_4": tf.reshape(self.cnn_train.gradients["conv_4"][0], [-1, 24 * 24 * 32, 1]) * tf.expand_dims(tf.gradients(self.train_loss, self.cnn_train.logits)[0], axis=1),
			"logits": tf.expand_dims(tf.gradients(self.train_loss, self.cnn_train.logits)[0], axis=2) * tf.expand_dims(tf.gradients(self.train_loss, self.cnn_train.logits)[0], axis=1),
		}
		
		self.train_values = train_values = {
			"conv_1": tf.reshape(MemoryValueModel(csn_gradients["conv_1"], self).outputs, [batch_size, -1, 27 * 27 * 32]),
			"conv_2": tf.reshape(MemoryValueModel(csn_gradients["conv_2"], self).outputs, [batch_size, -1, 26 * 26 * 32]),
			"conv_3": tf.reshape(MemoryValueModel(csn_gradients["conv_3"], self).outputs, [batch_size, -1, 25 * 25 * 32]),
			# "conv_4": tf.reshape(MemoryValueModel(csn_gradients["conv_4"], self).outputs, [batch_size, -1, 24 * 24 * 32]),
			"logits": tf.reshape(MemoryValueModel(csn_gradients["logits"], self).outputs, [batch_size, -1, num_classes]),
		}

		# Calculating Value for Test Key

		dotp = tf.matmul(test_keys, train_keys, transpose_b=True)
		self.attention_weights = attention_weights = tf.nn.softmax(dotp)
		csn = dict(zip(train_values.keys(), [tf.matmul(attention_weights, value) for value in train_values.values()]))
		self.csn = {
			"conv_0": None,
			"conv_1": tf.reshape(csn["conv_1"], [-1, 27, 27, 32]),
			"conv_2": tf.reshape(csn["conv_2"], [-1, 26, 26, 32]),
			"conv_3": tf.reshape(csn["conv_3"], [-1, 25, 25, 32]),
			# "conv_4": tf.reshape(csn["conv_4"], [-1, 24, 24, 32]),
			"logits": tf.reshape(csn["logits"], [-1, num_classes]),
		}

		# Finally, pass CSN values to adaCNNNet

		self.cnn_test = adaCNNNet(
			name="cnn", 
			inputs=self.test_inputs,
			layers=4, 
			output_dim=num_classes,
			parent=self,
			is_training=self.is_training,
			csn=self.csn,
		)

		self.test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.test_labels, logits=self.cnn_test.logits))
		
		self.optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.test_loss)

		self.test_predictions = tf.argmax(self.cnn_test.logits * self.logit_mask, axis=1)
		self.test_accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(self.test_labels, axis=1), predictions=self.test_predictions)
