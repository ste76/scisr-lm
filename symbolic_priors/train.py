""" Train and evaluation script to train the symbolic priors."""

import logging
from typing import Union, Any, List

from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state, checkpoints

import jax
import jax.numpy as jnp
import tensorflow as tf

import numpy as np
import optax

import tensorflow_datasets as tfds

import dataset_builder
import model
import symbolic_expr_priors
from exceptions import symbolic_exceptions
from torch.utils.tensorboard import SummaryWriter



def compute_multiclass_accuracy(logits, labels, config):
  """Computes multiclass accuracy. 

  Correct prediction will involve the ground-truth classes in the top predictions.
  Eg: labels: [0, 1, 0, 1, 1]
      logits: [0.54, 0.1, 0.1, 0.13, 0.13]
      Here the model has predicted class 0, 4, 5, but the gt is 1,4,5.
      So accuracy is 2.
  
  Args:
    logits: Probabilitic prediction by the model.
    labels: Grouth truth label.

  Returns:
    Jnp array of [1,] of accuracy value averaged for the batch.
  """
  class_accuracies = []
  logits = (logits > config.threshold).astype(dtype=np.int32)
  for i in range(config.num_symbols):  # As there are 14 unique symbols.
    class_accuracies.insert(i, np.mean(logits[:,i] == labels[:, i]))
  return class_accuracies

def compute_loss(logits, labels, alpha=0.25, gamma=0.25):
  """Computes loss for a multi-label classification problem."""
  epsilon = 1e-7
  labels = tf.cast(labels, tf.float32)
  logits = tf.cast(logits, tf.float32)

  alpha_t = labels*alpha + (tf.ones_like(labels)-labels)*(1-alpha)
  logits = tf.clip_by_value(logits, epsilon, 1. - epsilon)

  y_t = tf.multiply(logits, labels) + tf.multiply(1-logits, 1-labels)
  ce = -tf.math.log(y_t)
  weight = tf.pow(tf.subtract(1., y_t), gamma)
  fl1 = tf.multiply(weight, ce)
  fl2 = tf.linalg.matmul(fl1, tf.transpose(alpha_t))
  loss = tf.reduce_mean(fl2)
  return loss


@jax.jit
def apply_model(state, x, y, labels, threshold):
  """Computes gradient, loss and logits."""

  def loss_fn(params):
    # Step1: Get the model's prediction.
    logits = state.apply_fn({'params': params}, x, y)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
    return loss, logits

  # Step2: Compute gradient.
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)

  # Step3: Compute all label accuracy.
  logits = (logits > threshold).astype(dtype=jnp.int32)
  all_accuracy = jnp.mean(logits == labels)
  return grads, loss, logits, labels, all_accuracy


@jax.jit
def update_model(state, grads):
  """Updates the model with gradient computations."""
  return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng, config):
  """Runs training for a single epoch."""
  # train_ds_size = len(train_ds['x'])
  # steps_per_epoch = train_ds_size // batch_size

  # # Skips incomplete batch.
  # perms = jax.random.permutation(rng, len(train_ds['x']))
  # perms = perms[: steps_per_epoch * batch_size]
  # perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_all_accuracy = []
  epoch_class_accuracy = []

  for i, batch in enumerate(iter(train_ds)):
    batch_x = batch['x']
    batch_y = batch['y']
    batch_labels = batch['labels']
    # logging.info("batch_x:%s, batch_y:%s", batch_x, batch_y)

    # Step1: Compute gradients and lotss.
    grads, _, logits, labels, all_accuracy = apply_model(
      state, batch_x, batch_y, batch_labels, config.threshold)
    
    # Step2: Update the model with new gradients
    state = update_model(state, grads)
    
    # Step3: Compute accuracy outside the jax.jit.
    loss = compute_loss(logits, labels)
    class_accuracies = compute_multiclass_accuracy(logits, labels, config)
    
    # Step4: update loss and accuracy metrics for tracking.
    epoch_loss.append(loss)
    epoch_all_accuracy.append(all_accuracy)
    epoch_class_accuracy.append(class_accuracies)

    if i % 100 == 0:
      logging.info(
        'train_loss: %.4f, train_all_accuracy: %.2f, train_class_accuracy: %s'
        % (
            np.mean(epoch_loss),
            np.mean(epoch_all_accuracy) * 100,
            np.mean(epoch_class_accuracy, axis=0) * 100,
        )
      )

  train_loss = np.mean(epoch_loss)
  train_all_accuracy = np.mean(epoch_all_accuracy)
  train_class_accuracy = np.mean(epoch_class_accuracy, axis=0)

  return state, train_loss, train_all_accuracy, train_class_accuracy


def test_epoch(state, test_ds, batch_size, rng, config):
  """Runs training for a single epoch."""
  epoch_loss = []
  epoch_all_accuracy = []
  epoch_class_accuracy = []

  for batch in iter(test_ds):
    batch_x = batch['x']
    batch_y = batch['y']
    batch_labels = batch['labels']

    # Step1: Compute loss, not need to compute gradients or apply gradients.
    _, _, logits, labels, all_accuracy = apply_model(
      state, batch_x, batch_y, batch_labels, config.threshold)
    
    # Step3: Compute accuracy outside the jax.jit.
    loss = compute_loss(logits, labels)
    class_accuracies = compute_multiclass_accuracy(logits, labels, config)
    
    # Step4: update loss and accuracy metrics for tracking.
    epoch_loss.append(loss)
    epoch_all_accuracy.append(all_accuracy)
    epoch_class_accuracy.append(class_accuracies)

  test_loss = np.mean(epoch_loss)
  test_all_accuracy = np.mean(epoch_all_accuracy)
  test_class_accuracy = np.mean(epoch_class_accuracy, axis=0)

  return test_loss, test_all_accuracy, test_class_accuracy


def create_train_state(init_rng, config):
  """Creates initial train state to update the parameters."""
  symbol_prior_model = model.SymbolicPrior(
    num_symbols=config.num_symbols
  )
  params = symbol_prior_model.init(init_rng, jnp.ones([1]), jnp.ones([1]))["params"]
  # tx = optax.sgd(config.learning_rate, config.momentum)
  tx = optax.adam(config.learning_rate)
  return train_state.TrainState.create(apply_fn=symbol_prior_model.apply, params=params, tx=tx)


def get_datasets(config, train_ds_path, test_ds_path):
  """Get tensorflow train and test dataset."""
  try:
    train_ds = dataset_builder.create_dataset_builder(
        file_pattern=train_ds_path,
        batch_size=config.batch_size,
        is_training=True
    )
  except Exception as e:
    msg = f"Valid train dataset should be provided, path not found: {train_ds_path}, {e}"
    logging.error(msg)
    raise symbolic_exceptions.SymbolicPriorException(msg)  

  try:
    test_ds = dataset_builder.create_dataset_builder(
        file_pattern=test_ds_path,
        batch_size=config.batch_size,
        is_training=True
    )
  except Exception as e:
    msg = f"Valid train dataset should be provided, path not found: {test_ds_path}, {e}"
    logging.error(msg)
    raise symbolic_exceptions.SymbolicPriorException(msg)  

  batch_cnt = 0
  for batch in train_ds.as_numpy_iterator():
    batch_cnt += 1
  logging.info('Number of batchs in an epoch for training: %s based on batch size: %s', 
    batch_cnt, config.batch_size)
  
  return train_ds.as_numpy_iterator(), test_ds.as_numpy_iterator()


def train_and_evaluate(config, **kwargs) -> train_state.TrainState:
  """Runs training and evaluation of the symbolic prior model.
  
  Args:
    config: Hyperparamterconfiguration.
    workdir: Working directory to store tensorboard graphs.
    kwargs: Additional parameters such as dataset path, dataset related variables.

  Returns:
    The train state that includes the .params.
  """
  train_ds_path = kwargs.get('train_ds_path', None)
  test_ds_path = kwargs.get('test_ds_path', None)
  workdir = kwargs.get('workdir', './')
  checkpoint_dir = kwargs.get('checkpoint_dir', './')

  train_ds, test_ds = get_datasets(config, train_ds_path, test_ds_path)
  rng = jax.random.key(0)

  # Following are the tensorboard related settings.
  summary_writer = SummaryWriter(workdir)
  # summary_writer.add_hparams(vars(config))

  classind2symbol = symbolic_expr_priors.ALL_CLASS_IND_SYMBOLIC_PRIMITIVE_DICT
  summary_writer_layout = {}
  for i in range(config.num_symbols):
    symb = f'symbol-{classind2symbol[i]}'
    summary_writer_layout[symb] = ["Multiline", [f"train/{symb}_accuracy", f"test/{symb}_accuracy"]]
  
  summary_writer_layout["loss"] = ["Multiline", ["loss/train", "loss/validation"]]
  summary_writer_layout["all_accuracy"] = ["Multiline", ["all_accuracy/train", "all_accuracy/validation"]]
  layout = {
    "Symbolic Prior Pre-training": summary_writer_layout,
  }
  summary_writer.add_custom_scalars(layout)

  # Creation of training state.
  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, config)
  
  for epoch in range(1, config.num_epochs + 1):
    train_ds, test_ds = get_datasets(config, train_ds_path, test_ds_path)
    rng, input_rng = jax.random.split(rng)
    
    # Run training for one full epoch.
    state, train_loss, train_all_accuracy, train_class_accuracy = train_epoch(
      state, train_ds, config.batch_size, input_rng, config)

    # Evaluate the model for each epoch.
    test_loss, test_all_accuracy, test_class_accuracy = test_epoch(
      state, test_ds, config.batch_size, input_rng, config)

    logging.info(
        'epoch:% 3d, '
        'train_loss: %.4f, train_all_accuracy: %.2f, train_class_accuracy: %s'
        'test_loss: %.4f, test_all_accuracy: %.2f, test_class_accuracy: %s'
        % (
            epoch,
            train_loss,
            train_all_accuracy * 100,
            train_class_accuracy * 100,
            test_loss,
            test_all_accuracy * 100,
            test_class_accuracy * 100,
        )
    )

    # Write the tensorboard summary.
    summary_writer.add_scalar("loss/train", train_loss, epoch)
    summary_writer.add_scalar("loss/validation", test_loss, epoch)
  
    summary_writer.add_scalar("all_accuracy/train", train_all_accuracy, epoch)
    summary_writer.add_scalar("all_accuracy/validation", test_all_accuracy, epoch)

    for i in range(config.num_symbols):
      symb = f'symbol-{classind2symbol[i]}'

      summary_writer.add_scalar(f"train/{symb}_accuracy", train_class_accuracy[i], epoch)
      summary_writer.add_scalar(f"test/{symb}_accuracy", test_class_accuracy[i], epoch)
    
    # Write the check point.
    checkpoints.save_checkpoint(ckpt_dir=checkpoint_dir, target=state, step=epoch)

  summary_writer.flush()
  return state