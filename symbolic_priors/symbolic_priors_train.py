r"""Training script to train symbolic_priors regression model.

Usage:
  blaze run -c opt \
    learning/gemini/gemax/experimental/symbolic_llm/symbolic_priors:symbolic_priors_train

"""

from collections.abc import Sequence
from absl import app
from absl import flags

import jax
import jax.numpy as jnp

from google3.learning.gemini.gemax.experimental.symbolic_llm.symbolic_priors import dataset_builder


_BATCH_SIZE = flags.DEFINE_integer(
    name='batch_size',
    default=10,
    help='Training batch size.',
)

_SRC_FILES = flags.DEFINE_string(
    name='src_files',
    # default='/cns/uy-d/home/sthoppay/exp/ttl=5y/symbolic_llm/prior_dataset/priors_expression_20241116_205454/1/expr.sst*',
    default='/cns/uy-d/home/sthoppay/exp/ttl=5y/symbolic_llm/prior_dataset/priors_expression_20241117_144716/1/expr.sst*',
    help='Input SSTable containing training data.',
)

_IS_TRAIN = flags.DEFINE_bool(
    name='is_training',
    default=True,
    help='Is the run for train or test.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  dataset = dataset_builder.create_dataset_builder(
      file_pattern=_SRC_FILES.value,
      batch_size=_BATCH_SIZE.value,
      is_training=_IS_TRAIN.value
  )
  for batch in dataset:
    for k, v in batch.items():
      # print('batch size: ', jax.tree_util.tree_map(v, jnp.shape))
      print('k: ', k, ' val: ', v.shape)


if __name__ == '__main__':
  app.run(main)
