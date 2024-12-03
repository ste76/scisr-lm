"""Builds tensorflow dataset to train symbolic_priors regression model."""

from typing import Any, Callable, List, Union

import tensorflow.google as tf
from google3.learning.gemini.gemax.experimental.symbolic_llm.symbolic_priors import symbolic_expr_priors


def parse_fn(proto_key: bytes, proto: bytes):
  """Function to process the tf.train.Examples into batch data."""
  del proto_key

  features = {
      'x': tf.io.FixedLenFeature([1], tf.float32),
      'y': tf.io.FixedLenFeature([1], tf.float32),
      'classes': tf.io.VarLenFeature(tf.string),
  }
  feature_val = tf.io.parse_single_example(proto, features)
  sample_features = {
      'x': tf.convert_to_tensor(features['x'][0]),
      'y': tf.convert_to_tensor(features['y'][0])
      }
  # label_idx = [
  #     symbolic_expr_priors.ALL_SYMBOLIC_PRIMITIVE_CLASS_DICT[idx]
  #     for idx in tf.ragged.constant(feature_val['classes'])
  # ]
  # class_len = len(
  #     symbolic_expr_priors.ALL_SYMBOLIC_PRIMITIVE_CLASS_DICT.keys())
  # labels = tf.zeros(class_len)
  # labels[label_idx] = 1
  labels = tf.sparse.to_dense(feature_val['classes'])
  sequence_length = 10
  labels = labels.to_tensor(shape=(labels.bounding_shape()[0], sequence_length))
  # sample_labels = {'labels': tf.sparse.to_dense(feature_val['classes'])}
  sample_labels = {'labels': labels}
  return sample_labels


def create_dataset_builder(
    file_pattern: Union[str, List[str]],
    map_fn: Callable[..., Any] = parse_fn,
    batch_size: int = 32,
    is_training: bool = False,
) -> tf.data.Dataset:
  """Creates tensorflow dataset to train symbolic_priors model."""
  files = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)

  dataset = files.interleave(
      tf.data.SSTableDataset,
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
