"""
Implementation of Conditionally-Shifted Neurons in Tensorflow

TODO:
- Implement adaResNet
- Add dropout

"""


from __future__ import print_function
try:
	raw_input
except:
	raw_input = input


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from absl import flags
from absl import app


from models import adaCNNModel, adaResNetModel
from data_generator import DataGenerator


FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")

# Task parameters
# WIP only omniglot for now
# flags.DEFINE_string("datasource", "omniglot", "Omniglot or miniImagenet")
flags.DEFINE_integer("num_classes", 5, "Number of classes per task eg. 5-way refers to 5 classes")
flags.DEFINE_integer("num_shot_train", 1, "Number of training samples per class per task eg. 1-shot refers to 1 training sample per class")
flags.DEFINE_integer("num_shot_test", 1, "Number of test samples per class per task")

# Training parameters
flags.DEFINE_integer("metatrain_iterations", 40000, "Number of metatraining iterations")
flags.DEFINE_integer("meta_batch_size", 32, "Batchsize for metatraining")
flags.DEFINE_float("meta_lr", 0.0003, "Meta learning rate")
flags.DEFINE_integer("validate_every", 500, "Frequency for metavalidation and saving")
flags.DEFINE_string("savepath", "models/", "Path to save or load models")
flags.DEFINE_string("logdir", "logs/", "Path to save Tensorboard summaries")

# Testing parameters
flags.DEFINE_integer("num_test_classes", None, "Number of classes per test task, if different from training")

# Logging parameters
flags.DEFINE_integer("print_every", 100, "Frequency for printing training loss and accuracy")


def main(unused_args):

	if FLAGS.train:

		data_generator = DataGenerator(
			datasource='omniglot',
			num_classes=FLAGS.num_classes,
			num_samples_per_class=FLAGS.num_shot_train+FLAGS.num_shot_test,
			batch_size=FLAGS.meta_batch_size,
			test_set=False,
		)

		# Tensorflow queue for metatraining dataset
		# metatrain_image_tensor - (batch_size, num_classes * num_samples_per_class, 28 * 28)
		# metatrain_label_tensor - (batch_size, num_classes * num_samples_per_class, num_classes)
		metatrain_image_tensor, metatrain_label_tensor = data_generator.make_data_tensor(train=True)
		train_inputs = tf.slice(metatrain_image_tensor, [0, 0, 0], [-1, FLAGS.num_classes*FLAGS.num_shot_train, -1])
		test_inputs = tf.slice(metatrain_image_tensor, [0, FLAGS.num_classes*FLAGS.num_shot_train, 0], [-1, -1, -1])
		train_labels = tf.slice(metatrain_label_tensor, [0, 0, 0], [-1, FLAGS.num_classes*FLAGS.num_shot_train, -1])
		test_labels = tf.slice(metatrain_label_tensor, [0, FLAGS.num_classes*FLAGS.num_shot_train, 0], [-1, -1, -1])
		metatrain_input_tensors = {
			'train_inputs': train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'train_labels': train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			'test_inputs': test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			'test_labels': test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}

		# Tensorflow queue for metavalidation dataset
		metaval_image_tensor, metaval_label_tensor = data_generator.make_data_tensor(train=False)
		train_inputs = tf.slice(metaval_image_tensor, [0, 0, 0], [-1, FLAGS.num_classes*FLAGS.num_shot_train, -1])
		test_inputs = tf.slice(metaval_image_tensor, [0, FLAGS.num_classes*FLAGS.num_shot_train, 0], [-1, -1, -1])
		train_labels = tf.slice(metaval_label_tensor, [0, 0, 0], [-1, FLAGS.num_classes*FLAGS.num_shot_train, -1])
		test_labels = tf.slice(metaval_label_tensor, [0, FLAGS.num_classes*FLAGS.num_shot_train, 0], [-1, -1, -1])
		metaval_input_tensors = {
			'train_inputs': train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'train_labels': train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			'test_inputs': test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			'test_labels': test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}

		# Graphs for metatraining and metavalidation
		# using scope reuse=tf.AUTO_REUSE, not sure if this is the best way to do it

		model_metatrain = adaCNNModel("model", num_classes=FLAGS.num_classes, input_tensors=metatrain_input_tensors, lr=FLAGS.meta_lr, logdir=FLAGS.logdir, prefix="metatrain")
		# WIP adaResNet
		# model_metatrain = adaResNetModel("model", n=num_classes, input_tensors=input_tensors, logdir=FLAGS.logdir + "train")

		model_metaval = adaCNNModel("model", num_classes=FLAGS.num_classes, input_tensors=metaval_input_tensors, lr=FLAGS.meta_lr, logdir=FLAGS.logdir, prefix="metaval", is_training=model_metatrain.is_training)
		# WIP adaResNet
		# model_metaval = adaResNetModel("model", n=num_classes, input_tensors=input_tensors, logdir=FLAGS.logdir + "val", is_training=model_metatrain.is_training)

		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		tf.train.start_queue_runners()

		saved_metaval_loss = np.inf
		try:
			for step in np.arange(FLAGS.metatrain_iterations):
				metatrain_loss, metatrain_preaccuracy, metatrain_postaccuracy, metatrain_summary, _ = sess.run([model_metatrain.test_loss, model_metatrain.train_accuracy, model_metatrain.test_accuracy, model_metatrain.summary, model_metatrain.optimize], {model_metatrain.is_training: False})
				if step > 0 and step % FLAGS.print_every == 0:
					model_metatrain.writer.add_summary(metatrain_summary, step)
					print("Step #{} - Loss : {:.3f} - PreAcc : {:.3f} - PostAcc : {:.3f}".format(step, metatrain_loss, metatrain_preaccuracy, metatrain_postaccuracy))
				if step > 0 and (step % FLAGS.validate_every == 0 or step == (FLAGS.metatrain_iterations - 1)):
					if step == (FLAGS.metatrain_iterations - 1):
						print("Training complete!")
					metaval_loss, metaval_preaccuracy, metaval_postaccuracy, metaval_summary = sess.run([model_metaval.test_loss, model_metaval.train_accuracy, model_metaval.test_accuracy, model_metaval.summary], {model_metatrain.is_training: False})
					model_metaval.writer.add_summary(metaval_summary, step)
					print("Validation Results - Loss : {:.3f} - PreAcc : {:.3f} - PostAcc : {:.3f}".format(metaval_loss, metaval_preaccuracy, metaval_postaccuracy))
					if metaval_loss < saved_metaval_loss:
						saved_metaval_loss = metaval_loss
						model_metatrain.save(sess, FLAGS.savepath, global_step=step, verbose=True)
		# Catch Ctrl-C event and allow save option
		except KeyboardInterrupt:
			response = raw_input("\nSave latest model at Step #{}? (y/n)\n".format(step))
			if response == 'y':
				model_metatrain.save(sess, FLAGS.savepath, global_step=step, verbose=True)
			else:
				print("Latest model not saved.")

	if FLAGS.test:

		NUM_TEST_SAMPLES = 600

		num_test_classes = FLAGS.num_test_classes or FLAGS.num_classes

		data_generator = DataGenerator(
			datasource='omniglot',
			num_classes=num_test_classes,
			num_samples_per_class=FLAGS.num_shot_train+FLAGS.num_shot_test,
			batch_size=1, # use 1 for testing to calculate stdev and ci95
			test_set=True,
		)

		image_tensor, label_tensor = data_generator.make_data_tensor(train=False)

		train_inputs = tf.slice(image_tensor, [0, 0, 0], [-1, num_test_classes*FLAGS.num_shot_train, -1])
		test_inputs = tf.slice(image_tensor, [0, num_test_classes*FLAGS.num_shot_train, 0], [-1, -1, -1])
		train_labels = tf.slice(label_tensor, [0, 0, 0], [-1, num_test_classes*FLAGS.num_shot_train, -1])
		test_labels = tf.slice(label_tensor, [0, num_test_classes*FLAGS.num_shot_train, 0], [-1, -1, -1])
		input_tensors = {
			'train_inputs': train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'train_labels': train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			'test_inputs': test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			'test_labels': test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}

		model = adaCNNModel("model", num_classes=FLAGS.num_classes, input_tensors=input_tensors, logdir=None, num_test_classes=num_test_classes)

		sess = tf.InteractiveSession()
		model.load(sess, FLAGS.savepath, verbose=True)
		tf.train.start_queue_runners()

		accuracy_list = []

		for task in np.arange(NUM_TEST_SAMPLES):
			accuracy = sess.run(model.test_accuracy, {model.is_training: False})
			accuracy_list.append(accuracy)
			if task > 0 and task % 100 == 0:
				print("Metatested on {} tasks...".format(task))

		avg = np.mean(accuracy_list)
		stdev = np.std(accuracy_list)
		ci95 = 1.96 * stdev / np.sqrt(NUM_TEST_SAMPLES)

		print("\nEnd of Test!")
		print("Accuracy                : {:.3f}".format(avg))
		print("StdDev                  : {:.3f}".format(stdev))
		print("95% Confidence Interval : {:.3f}".format(ci95))


if __name__ == "__main__":
	app.run(main)
