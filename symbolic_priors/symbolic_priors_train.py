r"""Training script to train symbolic_priors regression model.

Usage:
  python symbolic_priors/symbolic_priors_train.py \ 
  --dataset_path='/Users/muruga/vishnu3/scisr-lm/datasets' 
  --batch_size=1

"""
import argparse
import os
import sys
import json
import logging

# Add additional search paths to include current path files.
print(sys.path)
basepath = os.getcwd()
sys.path.append(basepath)

from collections.abc import Sequence
from absl import app
from absl import flags

from flax.training import checkpoints

import jax
import jax.numpy as jnp
import tensorflow as tf
import numpy as np

from config import hparam_config

import train


def restore_ckpt(checkpoint_dir, config):
  """Restores the latest checkpoint from the checkpoint directory."""
  # Creation of training state.
  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)
  state = train.create_train_state(init_rng, config)

  restored_state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=state)
  return restored_state  

def main(args) -> None:
  """Main entry for training of the symbolic priors model."""
  # Prepare input variables for training and evaluation.
  config = hparam_config.get_symbolicprior_config()
  params = {}
  if not args.dataset_path:
    basepath = "/".join(os.getcwd().split('/')[:-1])
  else:
    basepath = args.dataset_path
  params['train_ds_path'] = os.path.join(basepath, 'train/symbolic_priors_1.tfrecord')
  params['test_ds_path'] = os.path.join(basepath, 'test/symbolic_priors_1.tfrecord')
  params['workdir'] = args.workdir
  params['checkpoint_dir'] = args.checkpoint_dir

  # Run training.
  train_state = train.train_and_evaluate(config=config, **params)

  # To restore a stored checkpoint and validate if the stored and retrieved are the same.
  restored_state = restore_ckpt(args.checkpoint_dir, config)
  assert jax.tree_util.tree_all(
    jax.tree_multimap(lambda x, y: (x == y).all(), train_state.params, restored_state.params))



if __name__ == "__main__":
  FORMAT = '[%(asctime)s-%(filename)s %(lineno)s]: %(funcName)s-%(message)s'
  logging.basicConfig(format=FORMAT, stream=sys.stdout, level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('-bs', '--batch_size', 
    dest='batch_size',
    default=4,
    type=int, help='Batch size for training the symbolic prior.')
  parser.add_argument('-od', '--dataset_path', 
    dest='dataset_path',
    default=None,
    type=str, help='Path to the tfrecord dataset to read and display examples')
  parser.add_argument('-wd', '--workdir',
    dest='workdir',
    default='./symbolic_pretrain_workdir',
    type=str, help='Work directory to store tensorboard data.')
  parser.add_argument('-cd', '--checkpoint_dir',
    dest='checkpoint_dir',
    default='/Users/muruga/vishnu3/scisr-lm/symbolic_pretrain_ckpt',
    type=str, help='Directory to store trained checkpoint of the model.')
  args = parser.parse_args()
  params = vars(args)

  logging.info('Input Arguments: %s\n', json.dumps(params, indent=2))

  main(args)