"""Neural network model to perform multiclass classification of symbolic prior."""

from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp

import numpy as np
import optax


class SymbolicPrior(nn.Module):
  """A simple SymbolicPrior model."""
  num_symbols:int = 14

  @nn.compact
  def __call__(self, x, y):
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_symbols)(x)
    return x
