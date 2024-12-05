"""Training config for scisr-lm."""

from argparse import Namespace

def get_symbolicprior_config():
  """Get the default hyperparameter configuration."""
  config = Namespace(**{})

  config.learning_rate = 0.001
  config.momentum = 0.9
  config.batch_size = 128
  config.num_epochs = 100
  config.num_symbols = 14
  config.threshold = 0.5  # Threshold for a positive prediction of symbol.
  return config


def metrics():
  return []