import numpy as np
import tensorflow as tf


class DataSet(object):
	def __init__(self, filename_data, filename_gender, nbdata, batchSize=128):
		self.nbdata = nbdata
		self.name = filename_data
		# taille des images 48*48 pixels en niveau de gris
		self.dim = 2304
		self.data = None
		self.label = None
		self.batchSize = batchSize
		self.curPos = 0

		f = open(filename_data, 'rb')
		self.data = np.empty([nbdata, self.dim], dtype=np.float32)
		for i in range(nbdata):
			self.data[i, :] = np.fromfile(f, dtype=np.uint8, count=self.dim)
		f.close()

		f = open(filename_gender, 'rb')
		self.label = np.empty([nbdata, 2], dtype=np.float32)
		for i in range(nbdata):
			self.label[i, :] = np.fromfile(f, dtype=np.float32, count=2)
		f.close()

		print('nb data = ', self.nbdata)
		self.data = (self.data - 128.0) / 256.0

		tmpdata = np.empty([1, self.dim], dtype=np.float32)
		tmplabel = np.empty([1, 2], dtype=np.float32)
		arr = np.arange(nbdata)
		np.random.shuffle(arr)
		tmpdata = self.data[arr[0], :]
		tmplabel = self.label[arr[0], :]
		for i in range(nbdata - 1):
			self.data[arr[i], :] = self.data[arr[i + 1], :]
			self.label[arr[i], :] = self.label[arr[i + 1], :]
		self.data[arr[nbdata - 1], :] = tmpdata
		self.label[arr[nbdata - 1], :] = tmplabel

	def NextTrainingBatch(self):
		if self.curPos + self.batchSize > self.nbdata:
			self.curPos = 0
		xs = self.data[self.curPos:self.curPos + self.batchSize, :]
		ys = self.label[self.curPos:self.curPos + self.batchSize, :]
		self.curPos += self.batchSize
		return xs, ys

	def mean_accuracy(self, model):
		acc = 0
		for i in range(0, self.nbdata, self.batchSize):
			curBatchSize = min(self.batchSize, self.nbdata - i)
			images = self.data[i:i+curBatchSize,:]
			labels = self.label[i:i+curBatchSize,:]
			y = model(images, False)
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			acc += accuracy * curBatchSize
		acc /= self.nbdata
		tf.summary.scalar('Accuracy %s'%self.name, acc)
		return acc

	def prediction(self, model, data):
		pred = model(data, False)
		return pred
    
    
    