from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import math_ops
import os.path
import math
import sys
from six.moves import xrange
import sqlite3
import time
import tensorflow as tf
import numpy as np
import os
from bigsetiterator import *

'''
database = sqlite3.connect('big_set.db')
db = database.cursor()
print("Opened database file")

db.execute('SELECT Priority01, Deadline01, Arg01, Period01, Number_of_Jobs01, Offset01, Priority02, Deadline02, Arg02, Period02, Number_of_Jobs02, Offset02, Priority03, Deadline03, Arg03, Period03, Number_of_Jobs03, Offset03, Priority04, Deadline04, Arg04, Period04, Number_of_Jobs04, Offset04, Priority05, Deadline05, Arg05, Period05, Number_of_Jobs05, Offset05 from Dataset')
data = db.fetchall()
db.execute('SELECT Exit_Value from Dataset')
label = db.fetchall()
print len(data[:5])
print data[0]
print len(data[0])
print label[1800]
print len(label)
print len(label[1800])

'''

class Config:

	def __init__(self, learning_rate = .0001, hidden_size = 32, batch_size = 32, max_epoch = 50):

		config_file = open('config.txt', 'w')

		self.learning_rate = learning_rate
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.max_epoch = max_epoch
		

		config_file.write("Learning rate " + str(self.learning_rate) + "\n")
		config_file.write("hidden size " + str(self.hidden_size) + "\n")
		config_file.write("Batch size " + str(self.batch_size) + "\n")
		config_file.write("Max Epochs" + str(self.max_epoch) + "\n")
		config_file.close() 

class run_model:
	
	def __init__(self, model, config = None):

		if config is None:
			config = Config()

		self.config = config
		self.model = model
		self.dataset = Dataset()
		
		
		
	def add_placeholders(self):

		self.input_placeholder = tf.placeholder(tf.float32, shape= (self.config.batch_size, 30), name = 'input')
		self.label_placeholder = tf.placeholder(tf.float32, shape= (self.config.batch_size, 1), name = 'label')
	
	def fill_feed_dict(self, inputs, labels, feed_previous = False):

		feed_dict = {
		self.input_placeholder		: inputs,
		self.label_placeholder		: labels 
		}
		
		return feed_dict
	
	def run_epoch(self, epoch_no, sess, fp = None):
		
		start_time = time.time()
		steps_per_epoch = int(math.ceil(float(self.dataset.datasets[0].num_samples)) / float(self.config.batch_size))
		total_loss = 0
		for step in xrange(steps_per_epoch):
			inputs, labels = self.dataset.next_batch(
				self.dataset.datasets[0], self.config.batch_size, True)		
			if fp is None:
				if (epoch_no > 5):
					feed_previous = True
				else:
					feed_previous = False
			else:
				feed_previous = fp
					
			feed_dict = self.fill_feed_dict(inputs, labels, feed_previous = True)
			_, loss_value, outputs	= sess.run([self.train_op, self.loss_ops, self.prob], feed_dict = feed_dict)
			total_loss += loss_value
			duration = time.time() - start_time
			print ('loss_value', loss_value, ' ', step)
			sys.stdout.flush()
			
			if (step + 1 == steps_per_epoch) or ((step + 1) % 5000 == 0):

				print('Step %d: Loss = %.2f'% (step, loss_value))
				sys.stdout.flush()
				
				print('Training Data Eval:')
				self.print_titles(sess, self.dataset.datasets[0])
					
				
				print('Step %d: loss = %.2f' % (step, loss_value))
				print('Validation Data Eval:')
				loss_value = self.do_eval(sess, self.dataset.datasets[2])
				self.print_titles(sess, self.dataset.datasets[2])
			   
				print('Test Data Eval:')
				loss_value = self.do_eval(sess, self.dataset.datasets[1])
				self.print_titles(sess,self.dataset.datasets[1], 2)
				print('Step %d: loss = %.2f' % (step, loss_value))

				self.print_titles_in_files(sess, self.dataset.datasets[0])
				self.print_titles_in_files(sess, self.dataset.datasets[1])
				self.print_titles_in_files(sess, self.dataset.datasets[2])
				sys.stdout.flush()
			return float(total_loss) / float(steps_per_epoch)	
			
	def do_eval(self, sess, data_set):
		
		steps_per_epoch = int(math.ceil(float(self.dataset.datasets[0].num_samples)) / float(self.config.batch_size))
		total_loss = 0
		for step in xrange(steps_per_epoch):
			inputs, labels = self.dataset.next_batch(
				self.dataset.datasets[0], self.config.batch_size, True)
		
			feed_dict = self.fill_feed_dict(inputs, labels, feed_previous = True)
			_, loss_value, outputs	= sess.run([self.train_op, self.loss_ops, self.prob], feed_dict = feed_dict)
			total_loss += loss_value		
			
		return float(total_loss) / float(steps_per_epoch)        	
			
			
	def run_training(self):
		
		with tf.Graph().as_default():
			
			#conf = tf.ConfigProto(device_count = {'GPU': 0})
			self.add_placeholders()
			self.prob	= self.model.inference(self.input_placeholder, self.config.hidden_size, self.config.batch_size)
			self.loss_ops	= self.model.loss_ops( self.prob, self.label_placeholder)
			self.train_op	= self.model.training( self.loss_ops, self.config.learning_rate)
			
			init = tf.global_variables_initializer()

			saver = tf.train.Saver()
			#sess = tf.Session(config = conf)
			sess = tf.Session()
			summary_writer = tf.summary.FileWriter('logs', sess.graph)

			if (os.path.exists('last_model')):
				saver.restore(sess, last_model)

			else:
				sess.run(init)
			best_val_loss = float('inf')
			best_val_epoch = 0
			for epoch in xrange(self.config.max_epoch):
				print ('Epoch: '+ str(epoch))
				start = time.time()

				train_loss = self.run_epoch(epoch, sess)
				valid_loss = self.do_eval(sess, self.dataset.datasets[2])

				print ('training loss:{}'.format(train_loss))
				print ('Validation loss:{}'.format(valid_loss))

				if (valid_loss<= best_val_loss):
					best_val_loss = valid_loss
					best_val_epoch = epoch 
					saver.save(sess, './best_model')

				if (epoch == self.config.max_epoch-1):
					saver.save(sess, './last_model')

				print ("Total time:{}".format(time.time() - start))

			saver.restore(sess, 'best_model')
			test_loss = self.do_eval(sess, self.dataset.datasets[1])
			print ("Test Loss:{}".format(test_loss))	
			self.print_titles_in_files(sess, self.dataset.datasets[1])
			self.print_titles_in_files(sess, self.dataset.datasets[2])
			
	def print_titles(self, sess, data_set):		
		
		inputs, labels = self.dataset.next_batch(
			self.dataset.datasets[0], self.config.batch_size, False)
			
		feed_dict = self.fill_feed_dict(inputs, labels, feed_previous = True)	
		
		output_prob = sess.run(self.prob, feed_dict = feed_dict)
		
		pred_output = []
				
		for prob in output_prob:		
			if (prob < 0.5):
				pred_output.append(-1)
			else:
				pred_output.append(1)
		for i in xrange(len(labels)):
			print('Predicted output is: '+ str(pred_output[i]))
			print('Actual output is: '+ str(labels[i]))	
		
	def print_titles_in_files(self, sess, data_set):	
		
		f1 = open(data_set.name +'_final_result', 'wb')
		
		inputs, labels = self.dataset.next_batch(
			self.dataset.datasets[0], self.config.batch_size, False)
			
		feed_dict = self.fill_feed_dict(inputs, labels, feed_previous = True)	
		
		output_prob = sess.run(self.prob, feed_dict = feed_dict)
		
		pred_output = []
				
		for prob in output_prob:		
			if (prob <= 0.5):
				pred_output.append(-1)
			else:
				pred_output.append(1)	
		for i in xrange(len(labels)):
			f1.write('Predicted output is: '+ str(pred_output[i]) + '\n')
			f1.write('Actual Output is : '+ str(labels[i]) + '\n') 	 	 
		
	
class Basic_model:
	
	def add_cell(self, hidden_size):
	
		cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
		self.cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)	
	
	def inference(self, inputs, hidden_size, batch_size):
		
		self.add_cell(hidden_size)
				
		
		probabilities = []
		loss = 0.0	
		initial_state = self.cell.zero_state(batch_size, dtype=tf.float32)
		rnn_outputs, rnn_states = self.cell(inputs, initial_state)
		logit =  tf.layers.dense(rnn_outputs, 1)
		print(logit.shape)
		prob = tf.sigmoid(logit)
		return prob
		
	def loss_ops(self, prob, labels):
		
		
		print(prob.get_shape())
		print(labels.get_shape())
		loss = tf.reduce_mean(tf.losses.log_loss(
		            labels,
		            prob,
		            weights=1.0,
		            scope=None,
		            loss_collection=tf.GraphKeys.LOSSES,
		            #reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
		            ))	
		'''            
		prob.
		loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(p, l) for p, l in zip(math_ops.to_float(prob), labels)]	
		'''
		self.loss = loss
		
		return loss
	
	def training(self, loss, learning_rate):
		
		optimizer = tf.train.AdamOptimizer(learning_rate)
		train_op = optimizer.minimize(loss)
		return train_op
	
	
	
def main():
	
	runModel = run_model(Basic_model())
	runModel.run_training()


if __name__ == '__main__':
	main()			
	
	
	
	
	
	
	
		
			
					
