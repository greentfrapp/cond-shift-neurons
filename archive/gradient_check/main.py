"""
Implementation of Conditionally-Shifted Neurons in Tensorflow
"""

import tensorflow as tf
import numpy as np
from absl import flags
from absl import app

from utils import MiniImageNetModel
from utils import update_target_graph

FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")

# Training parameters
flags.DEFINE_string("savepath", "models/", "Path to save or load models")

def CSNModel(object):

	def __init__(self):
		super(CSNModel, self).__init__()
		self.dummy = MiniImageNetModel(k)
		self.model = MiniImageNetModel(k)
		# self.memory_function = MemoryFunction()

		""" Pseudocode for meta-training (this should be connected as a graph)
		for each task:
			csn_memory = np.array()
			for each sample in task_training_set:
				sample_csn = sess.run(self.dummy.csn_gradients, feed_dict)
				sample_csn_key = self.memory_key_function(sample)
				sample_csn_value = self.memory_value_function(sample_csn)
				csn_memory.update(sample_csn_key, sample_csn_value)
			test_key = self.memory_key_function(test_sample)
			csn_attention = csn_memory.align(test_key)
			test_value = csn_memory.retrieve(csn_attention)
			
			feed_dict = {
				self.model.csn = test_value,
				self.model.inputs = test_sample,
				self.model.labels = test_label,
			}
			loss, _ = sess.run([self.model.loss, self.model.optimize], feed_dict)

		"""


def main(unused_args):
	
	if FLAGS.train:
		from keras.datasets import mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		dummy = MiniImageNetModel("dummy", k=10)
		model = MiniImageNetModel("model", k=10, csn=dummy.csn_gradients)
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		copy_op = update_target_graph("dummy", "model")
		sess.run(copy_op)

		n_steps = 1000
		batchsize = 64
		rand_x = np.random.RandomState(1)
		rand_y = np.random.RandomState(1)
		start = 0
		loss = []
		for i in np.arange(n_steps):
			if (i + 1) % 50 == 0:
				print("Running Step #{}".format(i + 1))
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
				dummy.inputs: minibatch_x,
				dummy.labels: minibatch_y,
				dummy.is_training: False,
			}

			dummy_loss = sess.run(dummy.loss, feed_dict)
			
			feed_dict = {
				dummy.inputs: minibatch_x,
				dummy.labels: minibatch_y,
				dummy.is_training: False,
				model.inputs: minibatch_x,
				model.labels: minibatch_y,
				model.is_training: False,
			}

			model_loss = sess.run(model.loss, feed_dict)

			print(dummy_loss - model_loss)

			loss.append(dummy_loss - model_loss)

			if i == 50:
				print(np.mean(loss))
				quit()

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
