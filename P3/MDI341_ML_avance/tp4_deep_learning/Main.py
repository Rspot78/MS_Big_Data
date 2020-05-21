
import tensorflow as tf
import DataSets as ds
import Layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LoadModel = True

experiment_size = 10
train = ds.DataSet('../DataBases/data_%dk.bin'%experiment_size,'../DataBases/gender_%dk.bin'%experiment_size,1000*experiment_size)
test = ds.DataSet('../DataBases/data_test10k.bin','../DataBases/gender_test10k.bin',10000)

class ConvNeuralNet(tf.Module):
    def __init__(self):
        self.unflat = Layers.unflat('unflat',48, 48, 1)
        self.cv1 = Layers.conv('conv_1', output_dim=3, filterSize=3, stride=1)
        self.mp = Layers.maxpool('pool', 2)
        self.cv2 = Layers.conv('conv_2', output_dim=6, filterSize=3, stride=1)
        self.cv3 = Layers.conv('conv_3', output_dim=12, filterSize=3, stride=1)
        self.flat = Layers.flat()
        self.fc = Layers.fc('fc', 2)

    def __call__(self, x, log_summary):
        x = self.unflat(x, log_summary)
        x = self.cv1(x, log_summary)
        x = self.mp(x)
        x = self.cv2(x, log_summary)
        x = self.mp(x)
        x = self.cv3(x, log_summary)
        x = self.mp(x)
        x = self.flat(x)
        x = self.fc(x, log_summary)
        return x


def train_one_iter(model, optimizer, image, label, log_summary):
    with tf.GradientTape() as tape:
        y = model(image,log_summary)
        y = tf.nn.log_softmax(y)
        diff = label * y
        loss = -tf.reduce_sum(diff)
        if log_summary:
            tf.summary.scalar('cross entropy', loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

print ("-----------------------------------------------------")
print ("----------------------- %dk -------------------------"%experiment_size)
print ("-----------------------------------------------------")

train_summary_writer = tf.summary.create_file_writer('logs %dk'%experiment_size)
optimizer = tf.optimizers.Adam(1e-3)
simple_cnn = ConvNeuralNet()

if LoadModel:
	ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=simple_cnn)
	ckpt.restore('./saved_model-1')


for iter in range(5000):
	tf.summary.experimental.set_step(iter)

	if iter % 500 == 0:
		with train_summary_writer.as_default():
			acc1 = train.mean_accuracy(simple_cnn) * 100
			acc2 = test.mean_accuracy(simple_cnn) * 100
			print("iter= %6d accuracy - train= %.2f%% - test= %.2f%%" % (iter, acc1, acc2))

	ima, lab = train.NextTrainingBatch()
	with train_summary_writer.as_default():
		loss = train_one_iter(simple_cnn, optimizer, ima, lab, iter % 10 == 0)

	if iter % 100 == 0:
		print("iter= %6d - loss= %f" % (iter, loss))

if not LoadModel:
	ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=simple_cnn)
	ckpt.save('./saved_model')