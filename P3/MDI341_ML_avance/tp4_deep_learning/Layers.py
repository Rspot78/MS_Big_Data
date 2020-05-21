

import tensorflow as tf

class fc(tf.Module):
	def __init__(self, name, output_dim):
		self.scope_name = name
		self.output_dim = output_dim
		self.b = tf.Variable(tf.constant(0.0, shape=[self.output_dim]))
	def __call__(self, x, log_summary):
		if not hasattr(self, 'w'):
			w_init = tf.random.truncated_normal([x.shape[1], self.output_dim], stddev=0.1)
			self.w = tf.Variable(w_init)
			print('build fc %s  %d => %d' % (self.scope_name,  x.shape[1], self.output_dim))

		if log_summary:
			with tf.name_scope(self.scope_name) as scope:
				tf.summary.scalar("mean w", tf.reduce_mean(self.w))
				tf.summary.scalar("max w", tf.reduce_max(self.w))
				tf.summary.histogram("w", self.w)
				tf.summary.scalar("mean b", tf.reduce_mean(self.b))
				tf.summary.scalar("max b", tf.reduce_max(self.b))
				tf.summary.histogram("b", self.b)
		return tf.matmul(x, self.w) + self.b

class conv(tf.Module):
	def __init__(self, name, output_dim, filterSize, stride):
		self.scope_name = name
		self.filterSize = filterSize
		self.output_dim = output_dim
		self.stride = stride
		self.b = tf.Variable(tf.constant(0.0, shape=[self.output_dim]))
	def __call__(self, x, log_summary):
		if not hasattr(self, 'w'):
			w_init = tf.random.truncated_normal([self.filterSize, self.filterSize, x.shape[3], self.output_dim], stddev=0.1)
			self.w = tf.Variable(w_init)
			print('build conv %s %dx%d  %d => %d'%(self.scope_name,self.filterSize,self.filterSize, x.shape[3], self.output_dim))
		if log_summary:
			with tf.name_scope(self.scope_name) as scope:
				tf.summary.scalar("mean w", tf.reduce_mean(self.w))
				tf.summary.scalar("max w", tf.reduce_max(self.w))
				tf.summary.histogram("w", self.w)
				tf.summary.scalar("mean b", tf.reduce_mean(self.b))
				tf.summary.scalar("max b", tf.reduce_max(self.b))
				tf.summary.histogram("b", self.b)
		x = tf.nn.conv2d(x, self.w, strides=[1, self.stride, self.stride, 1], padding='SAME') + self.b
		return tf.nn.relu(x)

class maxpool(tf.Module):
	def __init__(self, name, poolSize):
		self.scope_name = name
		self.poolSize = poolSize

	def __call__(self, x):
		return tf.nn.max_pool2d(x, ksize=(1, self.poolSize, self.poolSize, 1),
								strides=(1, self.poolSize, self.poolSize, 1), padding='SAME')

class flat(tf.Module):
	def __call__(self, x):
		inDimH = x.shape[1]
		inDimW = x.shape[2]
		inDimD = x.shape[3]
		return tf.reshape(x, [-1, inDimH * inDimW * inDimD])

class unflat(tf.Module):
	def __init__(self, name, outDimH, outDimW, outDimD):
		self.scope_name = name
		self.new_shape = [-1, outDimH, outDimW, outDimD]
		print('def unflat %s ? => %d %d %d' % (self.scope_name, outDimH, outDimW, outDimD))

	def __call__(self, x, log_summary):
		x = tf.reshape(x, self.new_shape)
		if log_summary:
			with tf.name_scope(self.scope_name) as scope:
				tf.summary.image('input', x, 5)
		return x
    