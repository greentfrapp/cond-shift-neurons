"""
models - utils
"""

import tensorflow as tf


# Base Model class with save and load methods
class BaseModel(object):

	def __init__(self):
		super(BaseModel, self).__init__()

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

# With the Direct Feedback implementation, this should act position-wise
# Take C-dim vector per position and output a scalar
class MemoryValueModel(object):

	def __init__(self, inputs, parent, name="memory_value"):
		super(MemoryValueModel, self).__init__()
		self.name = name
		self.inputs = inputs
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model()

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
