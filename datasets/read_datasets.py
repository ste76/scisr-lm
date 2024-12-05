"""Reads selected number of examples from a value tfrecord file.

Usage:
  python datasets/read_datasets.py --dataset_path='/Users/muruga/vishnu3/scisr-lm/datasets/train/symbolic_priors_1.tfrecord'
"""
import argparse
import logging
import json
import os
import random
import sys

import numpy as np
import tensorflow as tf


def read_dataset(params):
  """Reads input dataset and displays specific number of examples."""
  dataset_path = params["dataset_path"]
  if not os.path.exists(dataset_path):
    logging.error(f"Input dataset {dataset_path} not found.")
    return

  num_examples = params.get("num_examples", 10)

  # TFRecord Handling: https://www.tensorflow.org/tutorials/load_data/tfrecord#reading_a_tfrecord_file_in_python
  raw_dataset = tf.data.TFRecordDataset(dataset_path)
  for raw_record in raw_dataset.take(num_examples):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    logging.info('%s', example)


if __name__ == "__main__":
  FORMAT = '[%(asctime)s-%(filename)s %(lineno)s]: %(funcName)s-%(message)s'
  logging.basicConfig(format=FORMAT, stream=sys.stdout, level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('-od', '--dataset_path', 
    dest='dataset_path',
    default=None,
    type=str, help='Path to the tfrecord dataset to read and display examples')
  parser.add_argument('-ne', '--num_examples',
    dest='num_examples',
    default=10,
    type=int, help='Total number of examples to read & display from the dataset.')

  args = parser.parse_args()
  params = vars(args)

  logging.info('Input Arguments: %s\n', json.dumps(params, indent=2))

  read_dataset(params)