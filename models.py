"""

Models

NOTE: Changed inputs of self.outputs in ResBlock to output instead of maxpool

"""

import tensorflow as tf
import numpy as np

from utils import update_target_graph

# Base Model class with save and load methods
class Model(object):

	def __init__(self):
		super(Model, self).__init__()

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

# ResBlock
class ResBlock(object):

	def __init__(self, inputs, n_filters, name, is_training, csn=None):
		super(ResBlock, self).__init__()
		self.inputs = inputs
		self.name = name
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(n_filters, is_training, csn)

	def build_model(self, n_filters, is_training=False, csn=None):
		conv_1 = tf.layers.conv2d(
			inputs=self.inputs,
			filters=n_filters,
			kernel_size=(3, 3),
			padding="same",
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			activation=tf.nn.relu,
			name="conv_1",
			reuse=tf.AUTO_REUSE,
		)
		bn_1 = tf.contrib.layers.batch_norm(
			inputs=conv_1,
			scope="bn_1",
			reuse=tf.AUTO_REUSE,
			# is_training=is_training,
			is_training=False,
		)
		conv_2 = tf.layers.conv2d(
			inputs=bn_1,
			filters=n_filters,
			kernel_size=(3, 3),
			padding="same",
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			activation=tf.nn.relu,
			name="conv_2",
			reuse=tf.AUTO_REUSE,
		)
		bn_2 = tf.contrib.layers.batch_norm(
			inputs=conv_2,
			scope="bn_2",
			reuse=tf.AUTO_REUSE,
			# is_training=is_training,
			is_training=False,
		)
		conv_3 = tf.layers.conv2d(
			inputs=bn_2,
			filters=n_filters,
			kernel_size=(3, 3),
			padding="same",
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			activation=tf.nn.relu,
			name="conv_3",
			reuse=tf.AUTO_REUSE,
		)
		bn_3 = tf.contrib.layers.batch_norm(
			inputs=conv_3,
			scope="bn_3",
			reuse=tf.AUTO_REUSE,
			# is_training=is_training,
			is_training=False,
		)
		res_conv = tf.layers.conv2d(
			inputs=self.inputs,
			filters=n_filters,
			kernel_size=(1, 1),
			padding="same",
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			activation=None,
			name="res_conv",
			reuse=tf.AUTO_REUSE,
		)
		max_pool = tf.layers.max_pooling2d(
			inputs=bn_3+res_conv,
			pool_size=(2, 2),
			strides=(1, 1),
		)
		# seems like the gradient should be added prior to the relu
		if csn is not None:
			max_pool += csn[self.name]
		output = tf.nn.relu(max_pool)
		# if csn is not None:
		# 	output += tf.nn.relu(csn[self.name])
		self.outputs = tf.layers.dropout(
			inputs=output,
			rate=0.5,
			training=is_training,
		)
		self.gradients = tf.gradients(output, max_pool)

# MiniResNet
class MiniResNet(object):

	def __init__(self, inputs, n, name, parent, is_training, csn):
		super(MiniResNet, self).__init__()
		self.name = name
		self.inputs = inputs
		self.is_training = is_training
		self.csn = csn
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(n)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, parent.name + '/' + self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self, n):
		self.resblock_1 = ResBlock(self.inputs, 64, name="resblock_1", is_training=self.is_training)
		output = self.resblock_1.outputs
		self.resblock_2 = ResBlock(output, 96, name="resblock_2", is_training=self.is_training)
		output = self.resblock_2.outputs
		self.resblock_3 = ResBlock(output, 128, name="resblock_3", is_training=self.is_training, csn=self.csn)
		output = self.resblock_3.outputs
		self.resblock_4 = ResBlock(output, 256, name="resblock_4", is_training=self.is_training, csn=self.csn)
		output = self.resblock_4.outputs
		output = tf.layers.conv2d(
			inputs=output,
			filters=1024,
			kernel_size=(1, 1),
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			activation=tf.nn.relu,
			name="main_conv_1",
			reuse=tf.AUTO_REUSE,
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
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			activation=None,
			name="main_conv_2",
			reuse=tf.AUTO_REUSE,
		)
		output = tf.reshape(output, [-1, 19 * 19 * 384])
		output = tf.layers.dense(
			inputs=output,
			units=n,
			kernel_initializer=tf.contrib.layers.xavier_initializer(),
			activation=None,
			name="main_logits",
			reuse=tf.AUTO_REUSE,
		)
		self.logits = output
		if self.csn is not None:
			self.logits += self.csn["logits"]

# MiniImageNetModel comprising several ResBlocks
class MiniImageNetModel(Model):

	def __init__(self, name, k=5, memory=None, memory_key_model=None):
		super(MiniImageNetModel, self).__init__()
		self.name = name
		self.csn = None
		with tf.variable_scope(self.name):
			self.build_model(k, memory)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self, k, memory=None):
		self.inputs = tf.placeholder(
			shape=(None, 28, 28, 1),
			dtype=tf.float32,
			name="inputs",
		)
		self.labels = tf.placeholder(
			shape=(None, 10),
			dtype=tf.float32,
			name="labels",
		)
		self.is_training = tf.placeholder(
			shape=(None),
			dtype=tf.bool,
			name="is_training",
		)
		
		if memory is not None:
			self.memory_key_model =  MemoryKeyModel(tf.reshape(self.inputs, [-1, 28 * 28 * 1]), units=32, parent=self)
			self.update_memory_key_model = update_target_graph("dummy/memory_key", "model/memory_key")
			test_key = self.memory_key_model.outputs
			dotp = tf.matmul(test_key, memory["keys"], transpose_b=True)
			attention_weights = tf.nn.softmax(dotp)
			csn = dict(zip(memory["values"].keys(), [tf.matmul(attention_weights, value) for value in memory["values"].values()]))
			self.csn = {
				"resblock_3": tf.reshape(csn["resblock_3"], [-1, 25, 25, 128]),
				"resblock_4": tf.reshape(csn["resblock_4"], [-1, 24, 24, 256]),
				"logits": tf.reshape(csn["logits"], [-1, 10]),
			}

		resblock_1 = ResBlock(self.inputs, 64, name="resblock_1", is_training=self.is_training)
		output = resblock_1.outputs
		resblock_2 = ResBlock(output, 96, name="resblock_2", is_training=self.is_training)
		output = resblock_2.outputs
		resblock_3 = ResBlock(output, 128, name="resblock_3", is_training=self.is_training, csn=self.csn)
		output = resblock_3.outputs
		resblock_4 = ResBlock(output, 256, name="resblock_4", is_training=self.is_training, csn=self.csn)
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
		if self.csn is not None:
			self.logits += self.csn["logits"]
		self.predictions = tf.argmax(self.logits, axis=1)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
		self.optimize = tf.train.MomentumOptimizer(learning_rate=1e-5, momentum=0.9).minimize(self.loss)

		if memory is None:

			csn_gradients = {
				"resblock_3": tf.reshape(resblock_3.gradients[0], [-1, 25 * 25 * 128, 1]) * tf.expand_dims(tf.gradients(self.loss, self.logits)[0], axis=1),
				"resblock_4": tf.reshape(resblock_4.gradients[0], [-1, 24 * 24 * 256, 1]) * tf.expand_dims(tf.gradients(self.loss, self.logits)[0], axis=1),
				"logits": tf.expand_dims(tf.gradients(self.loss, self.logits)[0], axis=2) * tf.expand_dims(tf.gradients(self.loss, self.logits)[0], axis=1),
			}

			csn_gradients = tf.concat(list(csn_gradients.values()), axis=1)
			self.memory_value_model = MemoryValueModel(csn_gradients, self)
			csn_gradients = self.memory_value_model.outputs
			csn_gradients = tf.split(csn_gradients, [25 * 25 * 128, 24 * 24 * 256, 10], axis=1)
			# self.memory_values = {
			# 	"resblock_3": tf.reshape(csn_gradients[0], [-1, 25, 25, 128]),
			# 	"resblock_4": tf.reshape(csn_gradients[1], [-1, 24, 24, 256]),
			# 	"logits": tf.reshape(csn_gradients[2], [-1, 10]),
			# }
			self.memory_values = {
				"resblock_3": csn_gradients[0][:, :, 0],
				"resblock_4": csn_gradients[1][:, :, 0],
				"logits": csn_gradients[2][:, :, 0],
			}

			self.memory_key_model = MemoryKeyModel(tf.reshape(self.inputs, [-1, 28 * 28 * 1]), units=32, parent=self)
			self.memory_keys = self.memory_key_model.outputs
			self.memory = {
				"keys": self.memory_keys,
				"values": self.memory_values,
			}

# MiniImageNetModel comprising several ResBlocks
# v2 integrating both train(dummy) and test(model) models
class NewMiniImageNetModel(Model):

	def __init__(self, name, n=5, input_tensors=None, logdir=None, is_training=None):
		super(NewMiniImageNetModel, self).__init__()
		self.name = name
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(n, input_tensors, is_training)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=5)
			if logdir is not None:
				self.writer = tf.summary.FileWriter(logdir)

	def build_model(self, n, input_tensors=None, is_training=None):

		if input_tensors is None:
			self.train_inputs = tf.placeholder(
				shape=(None, 28, 28, 1),
				dtype=tf.float32,
				name="train_inputs",
			)
			self.train_labels = tf.placeholder(
				shape=(None, n),
				dtype=tf.float32,
				name="train_labels",
			)
			self.test_inputs = tf.placeholder(
				shape=(None, 28, 28, 1),
				dtype=tf.float32,
				name="test_inputs"
			)
			self.test_labels = tf.placeholder(
				shape=(None, n),
				dtype=tf.float32,
				name="test_labels",
			)

		else:
			self.train_inputs = tf.reshape(input_tensors['train_inputs'], [-1, 28, 28, 1])
			self.train_labels = tf.reshape(input_tensors['train_labels'], [-1, n])
			self.test_inputs = tf.reshape(input_tensors['test_inputs'], [-1, 28, 28, 1])
			self.test_labels = tf.reshape(input_tensors['test_labels'], [-1, n])

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

		# MiniResNet

		self.miniresnet_train = MiniResNet(self.train_inputs, n, "miniresnet", self, self.is_training, None)
		
		self.train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.train_labels, logits=self.miniresnet_train.logits))

		# CSN Memory Matrix

		# - Keys

		self.memory_key_model = MemoryKeyModel(tf.reshape(self.inputs, [-1, 28 * 28 * 1]), units=32, parent=self)
		keys = tf.split(
			self.memory_key_model.outputs,
			[tf.shape(self.train_inputs)[0], tf.shape(self.test_inputs)[0]],
			axis=0,
		)
		self.train_keys = train_keys = tf.reshape(keys[0], [batch_size, -1, 32])
		self.test_keys = test_keys = tf.reshape(keys[1], [batch_size, -1, 32])

		# - Values

		csn_gradients = {
			"resblock_3": tf.reshape(self.miniresnet_train.resblock_3.gradients[0], [-1, 25 * 25 * 128, 1]) * tf.expand_dims(tf.gradients(self.train_loss, self.miniresnet_train.logits)[0], axis=1),
			"resblock_4": tf.reshape(self.miniresnet_train.resblock_4.gradients[0], [-1, 24 * 24 * 256, 1]) * tf.expand_dims(tf.gradients(self.train_loss, self.miniresnet_train.logits)[0], axis=1),
			"logits": tf.expand_dims(tf.gradients(self.train_loss, self.miniresnet_train.logits)[0], axis=2) * tf.expand_dims(tf.gradients(self.train_loss, self.miniresnet_train.logits)[0], axis=1),
		}

		# csn_gradients = tf.concat(list(csn_gradients.values()), axis=1)
		self.memory_value_model_resblock_3 = MemoryValueModel(csn_gradients["resblock_3"], self)
		self.memory_value_model_resblock_4 = MemoryValueModel(csn_gradients["resblock_4"], self)
		self.memory_value_model_logits = MemoryValueModel(csn_gradients["logits"], self)
		# csn_gradients = self.memory_value_model.outputs
		# csn_gradients = tf.split(csn_gradients, [25 * 25 * 128, 24 * 24 * 256, n], axis=1)
		# train_values = {
		# 	"resblock_3": tf.reshape(csn_gradients[0][:, :, 0], [batch_size, -1, 25 * 25 * 128]),
		# 	"resblock_4": tf.reshape(csn_gradients[1][:, :, 0], [batch_size, -1, 24 * 24 * 256]),
		# 	"logits": tf.reshape(csn_gradients[2][:, :, 0], [batch_size, -1, n]),
		# }
		train_values = {
			"resblock_3": tf.reshape(self.memory_value_model_resblock_3.outputs, [batch_size, -1, 25 * 25 * 128]),
			"resblock_4": tf.reshape(self.memory_value_model_resblock_4.outputs, [batch_size, -1, 24 * 24 * 256]),
			"logits": tf.reshape(self.memory_value_model_logits.outputs, [batch_size, -1, n]),
		}

		# self.train_values = train_values = {
		# 	"resblock_3": -1e0 * tf.reshape(self.miniresnet_train.resblock_3.gradients[0], [-1, 25 * 25 * 128]),
		# 	"resblock_4": -1e0 * tf.reshape(self.miniresnet_train.resblock_4.gradients[0], [-1, 24 * 24 * 256]),
		# 	"logits": -1e0 * tf.gradients(self.train_loss, self.miniresnet_train.logits)[0],
		# }

		# Calculating Value for Test Key

		dotp = tf.matmul(test_keys, train_keys, transpose_b=True)
		self.attention_weights = attention_weights = tf.nn.softmax(dotp)
		csn = dict(zip(train_values.keys(), [tf.matmul(attention_weights, value) for value in train_values.values()]))
		self.csn = {
			"resblock_3": tf.reshape(csn["resblock_3"], [-1, 25, 25, 128]),
			"resblock_4": tf.reshape(csn["resblock_4"], [-1, 24, 24, 256]),
			"logits": tf.reshape(csn["logits"], [-1, n]),
		}

		# self.csn = {
		# 	"resblock_3": -1e0 * self.miniresnet_train.resblock_3.gradients[0][0],
		# 	"resblock_4": -1e0 * self.miniresnet_train.resblock_4.gradients[0][0],
		# 	"logits": -1e0 * tf.gradients(self.train_loss, self.miniresnet_train.logits)[0][0],
		# }

		# Finally, pass CSN values to MiniResNet

		self.miniresnet_test = MiniResNet(self.test_inputs, n, "miniresnet", self, self.is_training, self.csn)

		self.test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.test_labels, logits=self.miniresnet_test.logits))
		tf.summary.scalar('episode_test_loss', self.test_loss)

		self.optimize = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9).minimize(self.test_loss)

		self.test_predictions = tf.argmax(self.miniresnet_test.logits, axis=1)

		self.test_accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(self.test_labels, axis=1), predictions=self.test_predictions)
		tf.summary.scalar('episode_test_accuracy', self.test_accuracy)

		self.summary = tf.summary.merge_all()

# CNN used in adaCNN
class adaCNNNet(object):

	def __init__(self, name, inputs, n, parent, is_training, csn):
		super(adaCNNNet, self).__init__()
		self.name = name
		self.inputs = inputs
		self.is_training = is_training
		self.csn = csn
		self.gradients = dict()
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(n)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, parent.name + '/' + self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=5)

	def build_model(self, n):
		running_output = self.inputs
		for i in np.arange(5):
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
			inputs=tf.reshape(running_output, [-1, 23 * 23 * 32]),
			units=n,
			activation=None,
			name="logits",
			reuse=tf.AUTO_REUSE,
		)
		self.logits = self.output
		if self.csn is not None:
			self.logits += self.csn["logits"]
		# self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
		# self.gradients["logits"] = tf.gradients(self.loss, self.logits)

class adaCNNModel(Model):

	def __init__(self, name, num_classes=5, input_tensors=None, logdir=None, is_training=None):
		super(adaCNNModel, self).__init__()
		self.name = name
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(num_classes, input_tensors, is_training)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=5)
			if logdir is not None:
				self.writer = tf.summary.FileWriter(logdir)

	def build_model(self, n, input_tensors=None, is_training=None):

		if input_tensors is None:
			self.train_inputs = tf.placeholder(
				shape=(None, 28, 28, 1),
				dtype=tf.float32,
				name="train_inputs",
			)
			self.train_labels = tf.placeholder(
				shape=(None, n),
				dtype=tf.float32,
				name="train_labels",
			)
			self.test_inputs = tf.placeholder(
				shape=(None, 28, 28, 1),
				dtype=tf.float32,
				name="test_inputs"
			)
			self.test_labels = tf.placeholder(
				shape=(None, n),
				dtype=tf.float32,
				name="test_labels",
			)

		else:
			self.train_inputs = tf.reshape(input_tensors['train_inputs'], [-1, 28, 28, 1])
			self.train_labels = tf.reshape(input_tensors['train_labels'], [-1, n])
			self.test_inputs = tf.reshape(input_tensors['test_inputs'], [-1, 28, 28, 1])
			self.test_labels = tf.reshape(input_tensors['test_labels'], [-1, n])
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

		self.cnn_train = adaCNNNet("cnn", self.train_inputs, n, self, self.is_training, None)
		
		self.train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.train_labels, logits=self.cnn_train.logits))

		# CSN Memory Matrix

		# - Keys

		self.memory_key_model = adaCNNNet("key_model", self.inputs, 32, self, self.is_training, None)
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
			"conv_4": tf.reshape(self.cnn_train.gradients["conv_4"][0], [-1, 24 * 24 * 32, 1]) * tf.expand_dims(tf.gradients(self.train_loss, self.cnn_train.logits)[0], axis=1),
			"logits": tf.expand_dims(tf.gradients(self.train_loss, self.cnn_train.logits)[0], axis=2) * tf.expand_dims(tf.gradients(self.train_loss, self.cnn_train.logits)[0], axis=1),
		}
		
		train_values = {
			"conv_1": tf.reshape(MemoryValueModel(csn_gradients["conv_1"], self).outputs, [batch_size, -1, 27 * 27 * 32]),
			"conv_2": tf.reshape(MemoryValueModel(csn_gradients["conv_2"], self).outputs, [batch_size, -1, 26 * 26 * 32]),
			"conv_3": tf.reshape(MemoryValueModel(csn_gradients["conv_3"], self).outputs, [batch_size, -1, 25 * 25 * 32]),
			"conv_4": tf.reshape(MemoryValueModel(csn_gradients["conv_4"], self).outputs, [batch_size, -1, 24 * 24 * 32]),
			"logits": tf.reshape(MemoryValueModel(csn_gradients["logits"], self).outputs, [batch_size, -1, n]),
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
			"conv_4": tf.reshape(csn["conv_4"], [-1, 24, 24, 32]),
			"logits": tf.reshape(csn["logits"], [-1, n]),
		}

		# Finally, pass CSN values to MiniResNet

		self.cnn_test = adaCNNNet("cnn", self.test_inputs, n, self, self.is_training, self.csn)

		self.test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.test_labels, logits=self.cnn_test.logits))
		tf.summary.scalar('episode_test_loss', self.test_loss)

		self.optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.test_loss)

		self.test_predictions = tf.argmax(self.cnn_test.logits, axis=1)

		self.test_accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(self.test_labels, axis=1), predictions=self.test_predictions)
		tf.summary.scalar('episode_test_accuracy', self.test_accuracy)

		self.summary = tf.summary.merge_all()


# This should take in a data sample and output a key vector
class MemoryKeyModel(object):

	def __init__(self, inputs, units, parent, name="memory_key"):
		super(MemoryKeyModel, self).__init__()
		self.name = name
		self.inputs = inputs
		with tf.variable_scope(self.name):
			self.build_model(units)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, parent.name + '/' + self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self, units):

		outputs = tf.layers.dense(
			inputs=self.inputs,
			units=32,
			activation=tf.nn.relu,
		)
		outputs = tf.layers.dense(
			inputs=outputs,
			units=32,
			activation=tf.nn.relu,
		)
		self.outputs = tf.layers.dense(
			inputs=outputs,
			units=units,
			activation=None,
		)

# With the Direct Feedback implementation, this should act position-wise
# Take C-dim vector per position and output a scalar
class MemoryValueModel(object):

	def __init__(self, inputs, parent, name="memory_value"):
		super(MemoryValueModel, self).__init__()
		self.name = name
		self.inputs = inputs
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, parent.name + '/' + self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self):

		outputs = tf.layers.dense(
			inputs=self.inputs,
			units=32,
			activation=tf.nn.relu,
			reuse=tf.AUTO_REUSE,
			name="dense_1",
		)
		outputs = tf.layers.dense(
			inputs=outputs,
			units=32,
			activation=tf.nn.relu,
			reuse=tf.AUTO_REUSE,
			name="dense_2",
		)
		self.outputs = tf.layers.dense(
			inputs=outputs,
			units=1,
			activation=None,
			reuse=tf.AUTO_REUSE,
			name="output",
		)


