"""Builds tensorflow dataset to train symbolic_priors regression model."""

from typing import Any, Callable, List, Union

import tensorflow as tf


def parse_fn(proto: bytes):
  """Function to process the tf.train.Examples into batch data."""
  del proto_key

  features = {
      'x': tf.io.FixedLenFeature([1], tf.float32),
      'y': tf.io.FixedLenFeature([1], tf.float32),
      'function': tf.io.FixedLenFeature([1], tf.string),
      'classes': tf.io.VarLenFeature(tf.string),
      'class_indices': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
  }
  feature_val = tf.io.parse_single_example(proto, features)
  sample_features = {
      'x': tf.convert_to_tensor(feature_val['x']),
      'y': tf.convert_to_tensor(feature_val['y']),
      'function': tf.convert_to_tensor(feature_val['function']),
      'labels': tf.convert_to_tensor(feature_val['class_indices']),
      }
  return sample_features


def create_dataset_builder(
    file_pattern: Union[str, List[str]],
    map_fn: Callable[..., Any] = parse_fn,
    batch_size: int = 32,
    is_training: bool = False,
) -> tf.data.Dataset:
  """Creates tensorflow dataset to train symbolic_priors model."""
  files = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
  print(f'files: {files}')

  dataset = files.interleave(
      tf.data.TFRecordDataset,
      cycle_length=16,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )

  if is_training:
    dataset = dataset.shuffle(1024)
  dataset = dataset.map(
      map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size)

  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset
