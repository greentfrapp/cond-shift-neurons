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

from models import NewMiniImageNetModel, adaCNNModel
from data_generator import DataGenerator
from utils import update_target_graph
from tasks import MNISTFewShotTask

FLAGS = flags.FLAGS

flags.DEFINE_bool("train_mnist", False, "Train")
flags.DEFINE_bool("test_mnist", False, "Test")

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")

# Training parameters
flags.DEFINE_string("savepath", "models/", "Path to save or load models")
flags.DEFINE_string("logdir", "log/", "Path to save Tensorboard summaries")


def main(unused_args):

	if FLAGS.train:

		update_batch_size = 1
		num_classes = 5

		data_generator = DataGenerator(
			datasource='omniglot',
			num_classes=num_classes,
			num_samples_per_class=2,
			batch_size=5,
			test_set=False,
		)

		# samples - (batch_size, num_classes * num_samples_per_class, 28 * 28)
		# labels - (batch_size, num_classes * num_samples_per_class, num_classes)
		train_image_tensor, train_label_tensor = data_generator.make_data_tensor(train=True)
		# train_image_tensor, train_label_tensor = data_generator.make_data_tensor(train=False)

		train_inputs = tf.slice(train_image_tensor, [0,0,0], [-1,num_classes*update_batch_size, -1])
		test_inputs = tf.slice(train_image_tensor, [0,num_classes*update_batch_size, 0], [-1,-1,-1])
		train_labels = tf.slice(train_label_tensor, [0,0,0], [-1,num_classes*update_batch_size, -1])
		test_labels = tf.slice(train_label_tensor, [0,num_classes*update_batch_size, 0], [-1,-1,-1])
		input_tensors = {
			'train_inputs': train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'train_labels': train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			'test_inputs': test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			'test_labels': test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}

		model = adaCNNModel("model", num_classes=num_classes, input_tensors=input_tensors, logdir=None)
		# model = NewMiniImageNetModel("model", n=num_classes, input_tensors=input_tensors, logdir=FLAGS.logdir + "train")

		# Construct graph for validation
		val_image_tensor, val_label_tensor = data_generator.make_data_tensor(train=False)
		train_inputs = tf.slice(val_image_tensor, [0,0,0], [-1,num_classes*update_batch_size, -1])
		test_inputs = tf.slice(val_image_tensor, [0,num_classes*update_batch_size, 0], [-1,-1,-1])
		train_labels = tf.slice(val_label_tensor, [0,0,0], [-1,num_classes*update_batch_size, -1])
		test_labels = tf.slice(val_label_tensor, [0,num_classes*update_batch_size, 0], [-1,-1,-1])
		input_tensors = {
			'train_inputs': train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'train_labels': train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			'test_inputs': test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			'test_labels': test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}
		model_val = adaCNNModel("model", num_classes=num_classes, input_tensors=input_tensors, logdir=None, is_training=model.is_training)
		# model_val = NewMiniImageNetModel("model", n=num_classes, input_tensors=input_tensors, logdir=FLAGS.logdir + "val", is_training=model.is_training)

		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		tf.train.start_queue_runners()

		n_steps = 30000
		moving_avg_accuracy = 0.
		min_val_loss = np.inf
		for step in np.arange(n_steps):
			loss, _, accuracy, summary = sess.run([model.test_loss, model.optimize, model.test_accuracy, model.summary], {model.is_training: False})
			moving_avg_accuracy = 0.1 * accuracy + 0.9 * moving_avg_accuracy
			if step % 50 == 0:
				# model.writer.add_summary(summary, i)
				# Validation
				val_accuracy, val_loss, summary = sess.run([model_val.test_accuracy, model_val.test_loss, model_val.summary], {model.is_training: False})
				# model_val.writer.add_summary(summary, i)
				# accuracy = None
				# print("Task #{} - Loss : {:.3f} - Acc : {:.3f} - Val Acc : {:.3f}".format(i + 1, loss, moving_avg, accuracy))
				print("Step #{} - Loss : {:.3f} - Acc : {:.3f} - Val Loss : {:.3f} - Val Acc : {:.3f}".format(step, loss, moving_avg_accuracy, val_loss, val_accuracy))
				if val_loss < min_val_loss:
					min_val_loss = val_loss
					model.save(sess, FLAGS.savepath, global_step=step, verbose=True)

	if FLAGS.test:

		update_batch_size = 1
		num_classes = 5

		data_generator = DataGenerator(
			datasource='omniglot',
			num_classes=num_classes,
			num_samples_per_class=2,
			batch_size=5,
			test_set=True,
		)

		# samples - (batch_size, num_classes * num_samples_per_class, 28 * 28)
		# labels - (batch_size, num_classes * num_samples_per_class, num_classes)
		train_image_tensor, train_label_tensor = data_generator.make_data_tensor(train=False)

		train_inputs = tf.slice(train_image_tensor, [0,0,0], [-1,num_classes*update_batch_size, -1])
		test_inputs = tf.slice(train_image_tensor, [0,num_classes*update_batch_size, 0], [-1,-1,-1])
		train_labels = tf.slice(train_label_tensor, [0,0,0], [-1,num_classes*update_batch_size, -1])
		test_labels = tf.slice(train_label_tensor, [0,num_classes*update_batch_size, 0], [-1,-1,-1])
		input_tensors = {
			'train_inputs': train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'train_labels': train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			'test_inputs': test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			'test_labels': test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}

		model = adaCNNModel("model", num_classes=num_classes, input_tensors=input_tensors, logdir=None)

		sess = tf.InteractiveSession()
		model.load(sess, FLAGS.savepath, verbose=True)
		# tf.global_variables_initializer().run()
		tf.train.start_queue_runners()

		accuracy_list = []

		for batch in np.arange(120):
			accuracy = sess.run(model.test_accuracy, {model.is_training: False})
			print("Batch #{} - Test Acc : {:.3f}".format(batch, accuracy))
			accuracy_list.append(accuracy)

		print("\nEnd of Test - Mean Accuracy - {:.3f}".format(np.mean(accuracy_list)))
				

	if FLAGS.train_mnist:
		task = MNISTFewShotTask()
		model = NewMiniImageNetModel("model", n=3)
		sess = tf.Session()
		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())

		n_tasks = 20000
		moving_avg = 0.
		for i in np.arange(n_tasks):
			(x_train, y_train), (x_test, y_test) = task.next_task(k=3, test_size=5)
			feed_dict = {
				model.train_inputs: np.expand_dims(x_train, axis=3) / 255.,
				model.train_labels: np.eye(3)[np.array(y_train, dtype=np.int32)],
				model.test_inputs: np.expand_dims(x_test, axis=3) / 255.,
				model.test_labels: np.eye(3)[np.array(y_test, dtype=np.int32)],
				model.is_training: False,
			}

			# print(sess.run(model.csn, feed_dict)["logits"])
			# print(sess.run(model.miniresnet_train.logits, feed_dict))
			# print(sess.run(model.miniresnet_test.logits, feed_dict))
			# print(sess.run(model.miniresnet_train.logits, feed_dict))
			# print()
			# print(y_train)
			# print(sess.run(model.test_loss, feed_dict))

			# if i == 10:
			# 	quit()

			loss, _, accuracy = sess.run([model.test_loss, model.optimize, model.test_accuracy], feed_dict)
			# accuracy = np.sum(y_test == predictions) / len(y_test)
			# accuracy = np.sum(y_train == predictions) / len(y_train)

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
				accuracy = sess.run(model.test_accuracy, feed_dict)
				# predictions = sess.run(model.test_predictions, feed_dict)
				# accuracy = np.sum(y_test == predictions) / len(y_test)
				
				print("Task #{} - Loss : {:.3f} - Acc : {:.3f} - Test Acc : {:.3f}".format(i + 1, loss, moving_avg, accuracy))


	if FLAGS.test_mnist:
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
