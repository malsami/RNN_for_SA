import sqlite3
import numpy as np
import sys
import os.path
import tensorflow as tf

class Datatype:
	def __init__(self, name, input_data, label, num_samples):
		
		self.name = name
		self.input = input_data
		self.label = label
		self.num_samples = num_samples
		
	
class Dataset:
	def make_batch(self, data, batch_size, l, count):
		
		batch = []
		batch = data[count:count+batch_size]
		count += batch_size
		while (len(batch)<batch_size):
			batch.append(np.zeros(l, dtype = int))
			count = 0
			
		return batch, count
		
	def next_batch(self, dt, batch_size, c = True):
		
		if c is True:
			count = dt.global_count_train
		else:
			count = dt.global_count_test
		
		input_data, count1 = self.make_batch(dt.input, batch_size, 30, count)	
		output_data, _	 = self.make_batch(dt.labels, batch_size, 1, count)
		
		if (c == True): 
			dt.global_count_train = count1 % dt.num_samples
		else:
			dt.global_count_test = count1 % dt.num_samples
		
		return input_data, output_data	   
		
			
	
	
	def __init__(self, filename = 'big_set.db'):
		
		database = sqlite3.connect(filename)
		db = database.cursor()
		print("Opened database file")

		db.execute('SELECT Priority01, Deadline01, Arg01, Period01, Number_of_Jobs01, Offset01, Priority02, Deadline02, Arg02, Period02, Number_of_Jobs02, Offset02, Priority03, Deadline03, Arg03, Period03, Number_of_Jobs03, Offset03, Priority04, Deadline04, Arg04, Period04, Number_of_Jobs04, Offset04, Priority05, Deadline05, Arg05, Period05, Number_of_Jobs05, Offset05 from Dataset')
		data = db.fetchall()
		db.execute('SELECT Exit_Value from Dataset')
		label = db.fetchall()
		l = len(data)
		tr_l = int(l*0.8)
		te_l = int(l*0.1)
		self.datasets = {}
		self.datasets[0] = Datatype('train', data[:tr_l], label[:tr_l], len(label[:tr_l]))
		self.datasets[1] = Datatype('test', data[tr_l:tr_l+te_l], label[tr_l:tr_l+te_l], len(label[tr_l:tr_l+te_l]))
		self.datasets[2] = Datatype('val', data[tr_l+te_l:], label[tr_l+te_l:], len(label[tr_l+te_l:]))
	
	
def main():
	data = Dataset()
	print data.datasets[2].input
	print len(data.datasets[0].input)
	print len(data.datasets[2].label)		
		
if __name__ == '__main__':
	main()		
	
	
