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
import matplotlib.pyplot as plt
from absl import flags
from absl import app

from models import NewMiniImageNetModel, MemoryKeyModel, MemoryValueModel
from utils import update_target_graph
from tasks import MNISTFewShotTask

FLAGS = flags.FLAGS

flags.DEFINE_bool("train_mnist", False, "Train")

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

		# dummy = MiniImageNetModel("dummy", k=10)
		# model = MiniImageNetModel("model", k=10, memory=dummy.memory, memory_key_model=dummy.memory_key_model)
		model = NewMiniImageNetModel("model", k=10)
		# csn = tf.concat(list(dummy.csn_gradients.values()), axis=1)
		# memory_value_model = MemoryValueModel(dummy.csn_gradients["resblock_3"])
		# memory_value = memory_value_model.outputs
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		# copy_op = update_target_graph("dummy", "model")
		# sess.run(copy_op)

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
			# feed_dict = {
			# 	dummy.inputs: minibatch_x,
			# 	dummy.labels: np.eye(10)[minibatch_y],
			# 	dummy.is_training: True,
			# 	model.inputs: minibatch_x[:32],
			# 	model.labels: np.eye(10)[minibatch_y[:32]],
			# 	model.is_training: True,
			# }

			feed_dict = {
				model.train_inputs: minibatch_x,
				model.train_labels: np.eye(10)[minibatch_y],
				model.test_inputs: minibatch_x[:32],
				model.test_labels: np.eye(10)[minibatch_y[:32]],
				model.is_training: True,
			}

			print(sess.run(model.test_loss, feed_dict))

			# memory = sess.run(dummy.memory, feed_dict)
			# csn = sess.run(model.csn, feed_dict)
			
			# print(memory["keys"].shape)
			# print(memory["values"]["resblock_3"].shape)
			# print(memory["values"]["resblock_4"].shape)
			# print(memory["values"]["logits"].shape)

			# for key, value in csn.items():
			# 	print(value.shape)

			# print(sess.run([dummy.loss, model.loss, model.optimize], feed_dict))

			quit()

	if FLAGS.train_mnist:
		task = MNISTFewShotTask()
		model = NewMiniImageNetModel("model", n=3)
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		n_tasks = 20000
		moving_avg = 0.
		for i in np.arange(n_tasks):
			(x_train, y_train), (x_test, y_test) = task.next_task(k=1, test_size=5)
			feed_dict = {
				model.train_inputs: np.expand_dims(x_train, axis=3) / 255.,
				model.train_labels: np.eye(3)[np.array(y_train, dtype=np.int32)],
				model.test_inputs: np.expand_dims(x_test, axis=3) / 255.,
				model.test_labels: np.eye(3)[np.array(y_test, dtype=np.int32)],
				model.is_training: True,
			}
			loss, _, predictions = sess.run([model.test_loss, model.optimize, model.test_predictions], feed_dict)
			accuracy = np.sum(y_test == predictions) / len(y_test)

			moving_avg = 0.1 * accuracy + 0.9 * moving_avg

			if (i + 1) % 50 == 0:

				(x_train, y_train), (x_test, y_test) = task.next_task(k=1, test_size=50, metatest=True)
				feed_dict = {
					model.train_inputs: np.expand_dims(x_train, axis=3) / 255.,
					model.train_labels: np.eye(3)[np.array(y_train, dtype=np.int32)],
					model.test_inputs: np.expand_dims(x_test, axis=3) / 255.,
					model.test_labels: np.eye(3)[np.array(y_test, dtype=np.int32)],
					model.is_training: False,
				}
				predictions = sess.run(model.test_predictions, feed_dict)
				accuracy = np.sum(y_test == predictions) / len(y_test)
				# accuracy = None

				print("Task #{} - Loss : {} - Acc : {} - Test Acc : {}".format(i + 1, loss, moving_avg, accuracy))


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
