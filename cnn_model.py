"""An example of a CNN model. The data is randomly generated and consists of
images of image_size * image_size pixels, in gray scale (1 channel). The
images represent plots of some y(x) function governed by some parameters. Those
parameters represent the labels that the CNN is supposed to predict.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
import generate_data_cnn as gd

tf.logging.set_verbosity(tf.logging.INFO)


parser = argparse.ArgumentParser()


parser.add_argument('--train_steps', default=10000, type=int,
                    help='Total number of training steps.')
parser.add_argument('--batch_size', default=5, type=int, help='batch size')
parser.add_argument('--evaluations', default=10, type=int,
                    help='Total number of evaluations to perform during '
                         'training.')
parser.add_argument('--logs_per_training', default=5, type=int,
                    help='Number of logs in each training call, in between '
                         'evaluation calls. The config will have'
                         'log_step_count_steps = train_steps / batch_size / '
                         'evaluations / logs_per_training.')
parser.add_argument('--eval_steps', default=2, type=int,
                    help='number of eval steps')

parser.add_argument('--image_size', default=40, type=int,
                    help='Input image size in pixels.')
parser.add_argument('--lw', default=0.2, type=float,
                    help='Line width for the plot.')
parser.add_argument('--print_image', default=True, type=bool,
                    help='print the images')

parser.add_argument('--kernel_size', default=5, type=float,
                    help='Kernel size for the convolution layers.')
parser.add_argument('--l2', default=0.5, type=float,
                    help='l2_regularization scale')
parser.add_argument('--learning_rate', default=0.0005,
                    help='Learning rate', type=float)


def cnn_model_fn(features, labels, mode, params):
  """Model function."""
  print('---------- Mode:', mode.upper(), ' ----------')
  image_size = params['image_size']
  kernel_size = params['kernel_size']
  learning_rate = params['learning_rate']
  l2_regularization = params.get('l2_regularization')
  if l2_regularization:
    regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regularization)
  else:
    regularizer = None

  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels].
  # Images are image_size * image_size pixels, and have one color channel.
  input_layer = tf.reshape(
      features["image"], [-1, image_size, image_size, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, image_size, image_size, 1]
  # Output Tensor Shape: [batch_size, image_size, image_size, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[kernel_size, kernel_size],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, image_size, image_size, 32]
  # Output Tensor Shape: [batch_size, image_size/2, image_size/2, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, image_size/2, image_size/2, 32]
  # Output Tensor Shape: [batch_size, image_size/2, image_size/2, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[kernel_size, kernel_size],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, image_size/2, image_size/2, 64]
  # Output Tensor Shape: [batch_size, image_size/4, image_size/4, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, image_size/4, image_size/4, 64]
  # Output Tensor Shape: [batch_size, image_size/4 * image_size/4 * 64]
  pool2_flat = tf.reshape(
      pool2, [-1, image_size // 4 * image_size // 4 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, image_size/4, image_size/4 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                          activation=tf.nn.relu,
                          kernel_regularizer=regularizer)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 2]
  output_layer = tf.layers.dense(inputs=dropout, units=2,
                                 kernel_regularizer=regularizer)

  # Reshape the output layer to remove dimensions of size 1 and to return
  # predictions.
  predictions = tf.squeeze(output_layer)

  # PREDICT Mode
  if mode == tf.estimator.ModeKeys.PREDICT:
    # Split the prediction.
    y0, v0y = tf.split(predictions, 2)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={'y0': y0, 'v0y': v0y})

  # Stack the labels into one numpy array.
  stacked_labels = tf.stack([labels['y0'], labels['v0y']], axis=1)

  # TRAIN or EVAL mode
  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.mean_squared_error(
      labels=stacked_labels, predictions=predictions)

  # TRAIN mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate)
    if params.get('clip_gradients'):
      # Apply gradient clipper.
      gradients = optimizer.compute_gradients(loss)
      capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var
                          in gradients if grad is not None]
      train_op = optimizer.apply_gradients(
          capped_gradients, global_step=tf.train.get_global_step())
    else:
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op)

  # EVAL mode
  assert mode == tf.estimator.ModeKeys.EVAL

  return tf.estimator.EstimatorSpec(
      mode=mode,
      # Report sum of error for compatibility with pre-made estimators
      loss=loss)


def train_and_evaluate_model(args):
  """Train and evaluate the model."""
  print('Arguments:', args)
  if args.image_size % 4 != 0:
    raise ValueError(
        'image_size is %d but must be divisible by 4.' % args.image_size)

  params = {'image_size': args.image_size,
            'kernel_size': args.kernel_size,
            'l2_regularization': args.l2,
            'learning_rate': args.learning_rate,
            }

  # Train and evaluate the model
  log_step_count_steps = max(1, args.train_steps / args.batch_size /
                             args.evaluations // args.logs_per_training)
  estimator = tf.estimator.Estimator(
      model_fn=cnn_model_fn,
      params=params,
      config=tf.estimator.RunConfig(
          log_step_count_steps=log_step_count_steps
      )
  )

  # Train and evaluate the model. The training will happen for 'steps' training
  # steps, after which the input_fn no longer generates usage. Then it moves on
  # to evaluate the model, and performs 'eval_steps' steps. When it's done, it
  # returns to training, and so on. So by setting the training input_fn's steps
  # to be train_steps/evaluations, we ensure we've done the correct number of
  # evaluations.
  input_kwargs = {'lw': args.lw}
  train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: gd.input_fn(
          image_size=args.image_size,
          batch_size=args.batch_size,
          steps=args.train_steps // args.evaluations,
          **input_kwargs),
      max_steps=args.train_steps // args.batch_size)

  eval_spec = tf.estimator.EvalSpec(
      input_fn=lambda: gd.input_fn(
          image_size=args.image_size,
          batch_size=args.batch_size,
          steps=args.batch_size * args.eval_steps,
          **input_kwargs),
      steps=args.eval_steps
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  return estimator


def _mse(expected, predicted):
  """Mean square error, same as the loss."""
  return ((np.asarray(expected) - np.asarray(predicted)) ** 2).mean()


def make_predictions(args, estimator):
  """Run the model in prediction mode."""
  test_params = [{'y0': 1, 'v0y': 7}, {'y0': 3, 'v0y': -2}]
  test_size = len(test_params)
  input_kwargs = {'lw': args.lw}

  predict_results = list(estimator.predict(
      input_fn=lambda: gd.input_fn(
          image_size=args.image_size,
          steps=test_size,
          default_params=test_params,
          **input_kwargs)))

  # Pick a size small enough that the image fits nicely on the screen.
  print_image_size = 28
  test_dataset = gd.input_fn(image_size=print_image_size,
                             steps=len(test_params),
                             default_params=test_params,
                             **input_kwargs)
  test_elem = test_dataset.make_one_shot_iterator().get_next()

  predicted_dataset = gd.input_fn(image_size=print_image_size,
                                  steps=len(predict_results),
                                  default_params=predict_results,
                                  **input_kwargs)
  predicted_element = predicted_dataset.make_one_shot_iterator().get_next()

  print('Arguments:', args)
  with tf.Session() as sess:
    for test, prediction in zip(test_params, predict_results):
      test_data = sess.run(test_elem)
      predicted_data = sess.run(predicted_element)
      if args.print_image:
        print('Expected:')
        print(test_data[0]['image'])
        print()
        print('Predicted:')
        print(predicted_data[0]['image'])
        print()
      print('True input:', test)
      print('Prediction:', prediction)
      print('mse labels:',
            _mse(list(test.values()), list(prediction.values())))
      print('mse image:',
            _mse(test_data[0]['image'], predicted_data[0]['image']))
      print()
      print()


def main(argv):
  args = parser.parse_args(argv[1:])
  print('Arguments:', args)

  estimator = train_and_evaluate_model(args)
  make_predictions(args, estimator)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
