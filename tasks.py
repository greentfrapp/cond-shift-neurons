
import numpy as np
import tensorflow as tf

from data_generator import DataGenerator, OmniglotGenerator

mnist_imported = False

# MNIST Few-shot Task, just for testing purposes
# 3-way K-shot
class MNISTFewShotTask(object):

	def __init__(self, seed=42):
		super(MNISTFewShotTask, self).__init__()
		global mnist_imported
		if not mnist_imported:
			from keras.datasets import mnist
			mnist_imported = True
		(self.x, self.y), _ = mnist.load_data()

		self.metatrain_labels = [0, 1, 2, 3, 4, 5]
		self.metatest_labels = [6, 7, 8, 9]

		self.rand_x = np.random.RandomState(seed)
		self.rand_y = np.random.RandomState(seed)

	def next_task(self, k, test_size=1, metatest=False):
		if metatest:
			mnist_labels = np.random.choice(self.metatest_labels, 3, replace=False)
		else:
			mnist_labels = np.random.choice(self.metatrain_labels, 3, replace=False)

		x_train = y_train = x_test = y_test = None
		for i, mnist_label in enumerate(mnist_labels):
			samples_idx = np.random.choice(np.arange(len(self.x[np.where(self.y == mnist_label)])), k + test_size, replace=False)
			samples = self.x[np.where(self.y == mnist_label)][samples_idx]
			labels = np.ones(k + test_size) * i
			if x_train is None:
				x_train = samples[:-test_size]
				y_train = labels[:-test_size]
				x_test = samples[-test_size:]
				y_test = labels[-test_size:]
			else:
				x_train = np.concatenate([x_train, samples[:-test_size]], axis=0)
				y_train = np.concatenate([y_train, labels[:-test_size]], axis=0)
				x_test = np.concatenate([x_test, samples[-test_size:]], axis=0)
				y_test = np.concatenate([y_test, labels[-test_size:]], axis=0)

		self.rand_x.shuffle(x_train)
		self.rand_y.shuffle(y_train)
		self.rand_x.shuffle(x_test)
		self.rand_y.shuffle(y_test)

		return (x_train, y_train), (x_test[:test_size], y_test[:test_size])

class OmniglotTask(object):

	def __init__(self):
		super(OmniglotTask, self).__init__()
		self.data_generator = DataGenerator(
			datasource='omniglot',
			num_classes=5,
			num_samples_per_class=1,
			batch_size=1,
		)
		# self.data_generator.generate()
		samples, labels = self.data_generator.make_data_tensor()
		sess = tf.InteractiveSession()
		# print(sess.run(samples).shape)
		# print(sess.run(labels).shape)

		tf.global_variables_initializer().run()
		tf.train.start_queue_runners()

		dense = tf.layers.dense(
			inputs=samples,
			units=5,
			activation=None,
		)
		sess.run(tf.global_variables_initializer())
		print(sess.run(samples).shape)




if __name__ == "__main__":
	task = OmniglotTask()