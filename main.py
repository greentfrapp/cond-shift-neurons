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


from models import adaFFNModel, adaCNNModel, adaResNetModel
from data_generator import DataGenerator


FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")

# Task parameters
flags.DEFINE_string("datasource", "omniglot", "omniglot or sinusoid (miniimagenet WIP)")
flags.DEFINE_integer("num_classes", 5, "Number of classes per task eg. 5-way refers to 5 classes")
flags.DEFINE_integer("num_shot_train", None, "Number of training samples per class per task eg. 1-shot refers to 1 training sample per class")
flags.DEFINE_integer("num_shot_test", None, "Number of test samples per class per task")

# Training parameters
flags.DEFINE_integer("metatrain_iterations", None, "Number of metatraining iterations")
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

	if FLAGS.train and FLAGS.datasource == "omniglot":

		num_shot_train = FLAGS.num_shot_train or 1
		num_shot_test = FLAGS.num_shot_test or 1

		data_generator = DataGenerator(
			datasource="omniglot",
			num_classes=FLAGS.num_classes,
			num_samples_per_class=num_shot_train+num_shot_test,
			batch_size=FLAGS.meta_batch_size,
			test_set=False,
		)

		# Tensorflow queue for metatraining dataset
		# metatrain_image_tensor - (batch_size, num_classes * num_samples_per_class, 28 * 28)
		# metatrain_label_tensor - (batch_size, num_classes * num_samples_per_class, num_classes)
		metatrain_image_tensor, metatrain_label_tensor = data_generator.make_data_tensor(train=True)
		train_inputs = tf.slice(metatrain_image_tensor, [0, 0, 0], [-1, FLAGS.num_classes*num_shot_train, -1])
		test_inputs = tf.slice(metatrain_image_tensor, [0, FLAGS.num_classes*num_shot_train, 0], [-1, -1, -1])
		train_labels = tf.slice(metatrain_label_tensor, [0, 0, 0], [-1, FLAGS.num_classes*num_shot_train, -1])
		test_labels = tf.slice(metatrain_label_tensor, [0, FLAGS.num_classes*num_shot_train, 0], [-1, -1, -1])
		metatrain_input_tensors = {
			"train_inputs": train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			"train_labels": train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			"test_inputs": test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			"test_labels": test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}

		# Tensorflow queue for metavalidation dataset
		metaval_image_tensor, metaval_label_tensor = data_generator.make_data_tensor(train=False)
		train_inputs = tf.slice(metaval_image_tensor, [0, 0, 0], [-1, FLAGS.num_classes*num_shot_train, -1])
		test_inputs = tf.slice(metaval_image_tensor, [0, FLAGS.num_classes*num_shot_train, 0], [-1, -1, -1])
		train_labels = tf.slice(metaval_label_tensor, [0, 0, 0], [-1, FLAGS.num_classes*num_shot_train, -1])
		test_labels = tf.slice(metaval_label_tensor, [0, FLAGS.num_classes*num_shot_train, 0], [-1, -1, -1])
		metaval_input_tensors = {
			"train_inputs": train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			"train_labels": train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			"test_inputs": test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			"test_labels": test_labels, # batch_size, num_classes * update_batch_size, num_classes
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
		metatrain_iterations = FLAGS.metatrain_iterations or 40000
		try:
			for step in np.arange(metatrain_iterations):
				metatrain_loss, metatrain_preaccuracy, metatrain_postaccuracy, metatrain_summary, _ = sess.run([model_metatrain.test_loss, model_metatrain.train_accuracy, model_metatrain.test_accuracy, model_metatrain.summary, model_metatrain.optimize], {model_metatrain.is_training: False})
				if step > 0 and step % FLAGS.print_every == 0:
					model_metatrain.writer.add_summary(metatrain_summary, step)
					print("Step #{} - Loss : {:.3f} - PreAcc : {:.3f} - PostAcc : {:.3f}".format(step, metatrain_loss, metatrain_preaccuracy, metatrain_postaccuracy))
				if step > 0 and (step % FLAGS.validate_every == 0 or step == (metatrain_iterations - 1)):
					if step == (metatrain_iterations - 1):
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

	if FLAGS.test and FLAGS.datasource == "omniglot":

		NUM_TEST_SAMPLES = 600

		num_test_classes = FLAGS.num_test_classes or FLAGS.num_classes

		num_shot_train = FLAGS.num_shot_train or 1
		num_shot_test = FLAGS.num_shot_test or 1

		data_generator = DataGenerator(
			datasource="omniglot",
			num_classes=num_test_classes,
			num_samples_per_class=num_shot_train+num_shot_test,
			batch_size=1, # use 1 for testing to calculate stdev and ci95
			test_set=True,
		)

		image_tensor, label_tensor = data_generator.make_data_tensor(train=False)

		train_inputs = tf.slice(image_tensor, [0, 0, 0], [-1, num_test_classes*num_shot_train, -1])
		test_inputs = tf.slice(image_tensor, [0, num_test_classes*num_shot_train, 0], [-1, -1, -1])
		train_labels = tf.slice(label_tensor, [0, 0, 0], [-1, num_test_classes*num_shot_train, -1])
		test_labels = tf.slice(label_tensor, [0, num_test_classes*num_shot_train, 0], [-1, -1, -1])
		input_tensors = {
			"train_inputs": train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			"train_labels": train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			"test_inputs": test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			"test_labels": test_labels, # batch_size, num_classes * update_batch_size, num_classes
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

	if FLAGS.train and FLAGS.datasource == "sinusoid":

		num_shot_train = FLAGS.num_shot_train or 10
		num_shot_test = FLAGS.num_shot_test or 10

		data_generator = DataGenerator(
			datasource="sinusoid",
			num_classes=None,
			num_samples_per_class=num_shot_train+num_shot_test,
			batch_size=FLAGS.meta_batch_size,
			test_set=None,
		)

		model = adaFFNModel("model", lr=FLAGS.meta_lr, logdir=FLAGS.logdir, prefix="metatrain")
		
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		saved_loss = np.inf
		metatrain_iterations = FLAGS.metatrain_iterations or 20000
		try:
			for step in np.arange(metatrain_iterations):
				batch_x, batch_y, amp, phase = data_generator.generate()
				train_inputs = batch_x[:, :num_shot_train, :]
				train_labels = batch_y[:, :num_shot_train, :]
				test_inputs = batch_x[:, num_shot_train:, :]
				test_labels = batch_y[:, num_shot_train:, :]
				feed_dict = {
					model.train_inputs: train_inputs,
					model.train_labels: train_labels,
					model.test_inputs: test_inputs,
					model.test_labels: test_labels,
					model.amp: amp, # use amplitude to scale loss
				}
				metatrain_preloss, metatrain_postloss, metatrain_summary, _ = sess.run([model.train_loss, model.test_loss, model.summary, model.optimize], feed_dict)
				if step > 0 and step % FLAGS.print_every == 0:
					model.writer.add_summary(metatrain_summary, step)
					print("Step #{} - PreLoss : {:.3f} - PostLoss : {:.3f}".format(step, np.mean(metatrain_preloss), metatrain_postloss))
				if step > 0 and (step % FLAGS.validate_every == 0 or step == (metatrain_iterations - 1)):
					if step == (metatrain_iterations - 1):
						print("Training complete!")
					if metatrain_postloss < saved_loss:
						saved_loss = metatrain_postloss
						model.save(sess, FLAGS.savepath, global_step=step, verbose=True)
		# Catch Ctrl-C event and allow save option
		except KeyboardInterrupt:
			response = raw_input("\nSave latest model at Step #{}? (y/n)\n".format(step))
			if response == 'y':
				model.save(sess, FLAGS.savepath, global_step=step, verbose=True)
			else:
				print("Latest model not saved.")

	if FLAGS.test and FLAGS.datasource == "sinusoid":

		num_shot_train = FLAGS.num_shot_train or 10

		data_generator = DataGenerator(
			datasource="sinusoid",
			num_classes=None,
			num_samples_per_class=num_shot_train,
			batch_size=1,
			test_set=None,
		)

		model = adaFFNModel("model", lr=FLAGS.meta_lr, logdir=FLAGS.logdir, prefix="metatrain", num_train_samples=num_shot_train, num_test_samples=50)
		
		sess = tf.InteractiveSession()
		model.load(sess, FLAGS.savepath, verbose=True)

		train_inputs, train_labels, amp, phase = data_generator.generate()

		x = np.arange(-5., 5., 0.2)
		y = amp * np.sin(x - phase)

		feed_dict = {
			model.train_inputs: x.reshape(int(50/num_shot_train), -1, 1)
		}

		prepredictions = sess.run(model.train_predictions, feed_dict)

		feed_dict = {
			model.train_inputs: train_inputs,
			model.train_labels: train_labels,
			model.test_inputs: x.reshape(1, -1, 1),
		}

		postprediction = sess.run(model.test_predictions, feed_dict)

		fig, ax = plt.subplots()
		ax.plot(x, y, color="#2c3e50", linewidth=0.8, label="Truth")
		ax.scatter(train_inputs.reshape(-1), train_labels.reshape(-1), color="#2c3e50", label="Training Set")
		ax.plot(x, prepredictions.reshape(-1), color="#f39c12", label="Before Shift", linestyle=':')
		ax.plot(x, postprediction.reshape(-1), label="After Shift", color='#e74c3c', linestyle='--')
		ax.legend()
		plt.show()

if __name__ == "__main__":
	app.run(main)
