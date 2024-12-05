"""Creates simulation dataset to train symbolic priors model.


Usage:
  python symbolic_priors/simulate_symbolic_expr.py  --output_dir="./datasets" --variable_cnt=1 --num_examples=1000000 --data_type='train'
"""

import argparse
import logging
import json
import os
import random
import sys

import numpy as np
import tensorflow as tf

import constants
import symbolic_expr_priors



def _bytes_feature(value):
	"""Returns a bytes_list tf.Example feature list."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
	"""Returns a float_list tf.Example feature list."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
	"""Returns a int64_list tf.Example feature list."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(x, y, function_name, classes, class_indice_array):
	"""Given the input, output, symbolic_func, classes, serialize to tf.train.Example."""
	# Create tf.Train.Feature
	feature = {
	  'x': _float_feature([x]),
	  'y': _float_feature([y]),
	  'function': _bytes_feature([bytes(function_name, 'utf-8')]),
	  'classes': _bytes_feature(classes),
	  'class_indices': _int64_feature(class_indice_array)
	}

	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto.SerializeToString()

def generate_dataset(params):
	"""Generates dataset by simulating the symbolic priors expression."""
	output_dir = params.get('output_dir', './dataset')
	variable_cnt = params.get('variable_cnt', 1)
	num_examples = params.get('num_examples', 100)
	data_type = params.get('data_type', 'train')

	# Step1: Identify the expressions corresponding to the variable_cnt input
	#  parameters.
	if variable_cnt == 1:
		symbolic_expr_list = constants.SYMBOLIC_EXPR_PRIORS_ONE_VARIABLE
	elif variable_cnt == 2:
		symbolic_expr_list = constants.SYMBOLIC_EXPR_PRIORS_TWO_VARIABLE
	else:
		raise NotImplementedError

	# Preparing output write directory.
	write_dir = os.path.join(output_dir, data_type)
	try:
		if not os.path.exists(write_dir):
			os.makedirs(write_dir)
	except Exception as e:
		logging.error(f'dataset directory {write_dir} creation failed.')

	write_output = os.path.join(write_dir, f'symbolic_priors_{variable_cnt}.tfrecord')
	writer = tf.io.TFRecordWriter(write_output)

	for i in range(num_examples):
		if i%100 == 0:
			logging.info(f'Generated {i} examples.')
		expr_index = random.choice(range(len(symbolic_expr_list)))
		symbolic_expr_dict = symbolic_expr_list[expr_index]

		x_range = symbolic_expr_dict['x']

		classes = list(symbolic_expr_dict['classes'])
		classes_encoded = [bytes(str(sym), 'utf-8') for sym in classes]

		all_class_indices = symbolic_expr_priors.ALL_SYMBOLIC_PRIMITIVE_CLASS_DICT
		num_classes = len(all_class_indices)
		class_indices = [all_class_indices[sym] for sym in classes]
		class_indice_array = np.zeros(num_classes, dtype=np.int64)
		class_indice_array[class_indices] = 1

		symbolic_expr_func = symbolic_expr_dict['function']


		# Step2: Simulate the symbolic expression function based on the input range.
		x = random.uniform(*x_range)
		y = symbolic_expr_func(x)
		key = str(symbolic_expr_func)+'_'+str(x)+'_'+str(y)


		# Step3: Write TFRecord dataset in the output_dir
		example = serialize_example(
			x=x, y=y, function_name=symbolic_expr_func.__name__, 
			classes=classes_encoded, class_indice_array=class_indice_array)
		writer.write(example)



if __name__ == "__main__":
	FORMAT = '[%(asctime)s-%(filename)s %(lineno)s]: %(funcName)s-%(message)s'
	logging.basicConfig(format=FORMAT, stream=sys.stdout, level=logging.INFO)

	parser = argparse.ArgumentParser()
	parser.add_argument('-od', '--output_dir', 
		dest='output_dir',
		default='../datasets',
		type=str, help='Path to the output directory to store the dataset.')
	parser.add_argument('-vc', '--variable_cnt',
		dest='variable_cnt',
		default=1,
		type=int, help='Expressions with variable_cnt parameters to simulate.')
	parser.add_argument('-ne', '--num_examples',
		dest='num_examples',
		default=10,
		type=int, help='Total number of train examples and 1/10 test examples to generate.')
	parser.add_argument('-dt', '--data_type',
		dest='data_type',
		default='train',
		type=str, help='Type of dataset to be generated. One of train, test, valid')

	args = parser.parse_args()
	params = vars(args)

	logging.info('Input Arguments: %s\n', json.dumps(params, indent=2))

	generate_dataset(params)