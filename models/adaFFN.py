"""
models - adaFFN
"""

import tensorflow as tf

from models.utils import BaseModel


# Network used in adaFFN
class adaFFNNet(object):

	def __init__(self, name, inputs, hidden, output_dim, parent, csn):
		super(adaFFNNet, self).__init__()
		self.name = name
		self.inputs = inputs
		self.csn = csn
		self.gradients = dict()
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(hidden, output_dim)

	def build_model(self, hidden, output_dim):
		dense_1 = tf.layers.dense(
			inputs=self.inputs,
			units=hidden,
			activation=None,
			name="dense_1",
			reuse=tf.AUTO_REUSE,
		)
		if self.csn is not None:
			dense_1 += self.csn["dense_1"]
		relu_1 = tf.nn.relu(dense_1)
		self.gradients["dense_1"] = tf.gradients(relu_1, dense_1)
		dense_2 = tf.layers.dense(
			inputs=relu_1,
			units=hidden,
			activation=tf.nn.relu,
			name="dense_2",
			reuse=tf.AUTO_REUSE,
		)
		if self.csn is not None:
			dense_2 += self.csn["dense_1"]
		relu_2 = tf.nn.relu(dense_2)
		self.gradients["dense_2"] = tf.gradients(relu_2, dense_2)
		self.outputs = tf.layers.dense(
			inputs=relu_2,
			units=output_dim,
			activation=None,
			name="output",
			reuse=tf.AUTO_REUSE,
		)
		if self.csn is not None:
			self.outputs += self.csn["outputs"]

class adaFFNModel(BaseModel):

	def __init__(self, name, lr=1e-4, logdir=None, prefix='', num_train_samples=10, num_test_samples=10, vary=False):
		super(adaFFNModel, self).__init__()
		self.name = name
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(lr, num_train_samples, num_test_samples, vary=False)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)
		if logdir is not None:
			self.writer = tf.summary.FileWriter(logdir + prefix)
			self.summary = tf.summary.merge([
				tf.summary.scalar("loss", self.test_loss, family=prefix),
			])

	def build_model(self, lr=1e-4, num_train_samples=10, num_test_samples=10, vary=False):

		self.train_inputs = tf.placeholder(
			shape=(None, num_train_samples, 1),
			dtype=tf.float32,
			name="train_inputs",
		)
		self.train_labels = tf.placeholder(
			shape=(None, num_train_samples, 1),
			dtype=tf.float32,
			name="train_labels",
		)
		self.test_inputs = tf.placeholder(
			shape=(None, num_test_samples, 1),
			dtype=tf.float32,
			name="test_inputs"
		)
		self.test_labels = tf.placeholder(
			shape=(None, num_test_samples, 1),
			dtype=tf.float32,
			name="test_labels",
		)
		# use amplitude to scale loss
		self.amp = tf.placeholder(
			shape=(None),
			dtype=tf.float32,
			name="amplitude"
		)

		batch_size = tf.shape(self.train_inputs)[0]

		self.inputs = tf.concat([
			tf.reshape(self.train_inputs, [-1, 1]),
			tf.reshape(self.test_inputs, [-1, 1])],
			axis=0,
		)

		# CNN

		self.regressor_train = adaFFNNet(
			name="regressor", 
			inputs=self.train_inputs,
			hidden=40,
			output_dim=1,
			parent=self,
			csn=None,
		)
		
		# Need to calculate training loss per task
		self.train_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.train_labels, predictions=self.regressor_train.outputs, reduction=tf.losses.Reduction.NONE), axis=1)
		self.train_predictions = self.regressor_train.outputs

		# CSN Memory Matrix

		# - Keys

		self.memory_key_model = adaFFNNet(
			name="key_model",
			inputs=self.inputs,
			hidden=10,
			output_dim=4,
			parent=self,
			csn=None,
		)
		keys = tf.split(
			self.memory_key_model.outputs,
			[tf.shape(self.train_inputs)[0]*tf.shape(self.train_inputs)[1], tf.shape(self.test_inputs)[0]*tf.shape(self.test_inputs)[1]],
			axis=0,
		)
		self.train_keys = train_keys = tf.reshape(keys[0], [batch_size, -1, 4])
		self.test_keys = test_keys = tf.reshape(keys[1], [batch_size, -1, 4])

		# - Values

		csn_gradients = {
			"dense_1": tf.reshape(self.regressor_train.gradients["dense_1"][0] * tf.gradients(self.train_loss, self.regressor_train.outputs)[0], [-1, 40, 1]),
			"dense_2": tf.reshape(self.regressor_train.gradients["dense_2"][0] * tf.gradients(self.train_loss, self.regressor_train.outputs)[0], [-1, 40, 1]),
			"outputs": tf.gradients(self.train_loss, self.regressor_train.outputs)[0] * tf.gradients(self.train_loss, self.regressor_train.outputs)[0],
		}
		
		self.train_values = train_values = {
			"dense_1": tf.reshape(adaFFNNet("value_model", csn_gradients["dense_1"], 10, 1, self, None).outputs, [batch_size, -1, 40]),
			"dense_2": tf.reshape(adaFFNNet("value_model", csn_gradients["dense_2"], 10, 1, self, None).outputs, [batch_size, -1, 40]),
			"outputs": tf.reshape(adaFFNNet("value_model", csn_gradients["outputs"], 10, 1, self, None).outputs, [batch_size, -1, 1]),
		}

		# Calculating Value for Test Key

		dotp = tf.matmul(test_keys, train_keys, transpose_b=True)
		self.attention_weights = attention_weights = tf.nn.softmax(dotp)
		csn = dict(zip(train_values.keys(), [tf.matmul(attention_weights, value) for value in train_values.values()]))
		self.csn = {
			"dense_1": tf.reshape(csn["dense_1"], [batch_size, -1, 40]),
			"dense_2": tf.reshape(csn["dense_2"], [batch_size, -1, 40]),
			"outputs": tf.reshape(csn["outputs"], [batch_size, -1, 1]),
		}

		# Finally, pass CSN values to adaCNNNet

		self.regressor_test = adaFFNNet(
			name="regressor", 
			inputs=self.test_inputs,
			hidden=40,
			output_dim=1,
			parent=self,
			csn=self.csn,
		)

		self.test_loss = tf.losses.mean_squared_error(labels=self.test_labels / self.amp, predictions=self.regressor_test.outputs / self.amp)
		
		self.optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.test_loss)

		self.test_predictions = self.regressor_test.outputs
