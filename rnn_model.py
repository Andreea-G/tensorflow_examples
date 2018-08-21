"""An example of a sequence to sequence RNN model, without embedding. The data
is a randomly generated time series of one-dimensional values obeying some
mathematical formula. The input data is a the first part of the time series,
and the output is the second part.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import generate_data_rnn as gd
from tensorflow.python.framework import dtypes

tf.logging.set_verbosity(tf.logging.INFO)
np.set_printoptions(threshold=np.nan, linewidth=1000, precision=2)


parser = argparse.ArgumentParser()
parser.add_argument('--train_steps', default=100000, type=int,
                    help='Total number of training steps.')
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--evaluations', default=10, type=int,
                    help='Total number of evaluations to perform during '
                         'training.')
parser.add_argument('--logs_per_training', default=5, type=int,
                    help='Number of logs in each training call, in between '
                         'evaluation calls. The config will have'
                         'log_step_count_steps = train_steps / batch_size / '
                         'evaluations / logs_per_training.')
parser.add_argument('--eval_steps', default=1, type=int,
                    help='number of eval steps')

parser.add_argument('--time_steps', default=32, type=int,
                    help='Total number of time steps, for the combined input '
                         'and output.')
parser.add_argument('--output_size', default=4, type=int,
                    help='How many of those time steps we want the model to '
                         'predict. The rest are input.')

parser.add_argument('--l2', default=None, type=float,
                    help='l2_regularization scale')
parser.add_argument('--learning_rate', default=0.0005,
                    help='Learning rate', type=float)
parser.add_argument('--num_rnn_nodes', default=32,
                    help='Number of nodes in the RNN network', type=int)
parser.add_argument('--num_rnn_layers', default=5,
                    help='Number of layers in the RNN network', type=int)


def _get_batch_size(input):
  """The batch_size is the first element in the shape.

  Raises:
    AssertionError if the batch_size is not known.
  """
  batch_size = input.get_shape().as_list()[0]
  if batch_size is None:
    raise AssertionError('Batch size is not known.')
  return batch_size


def _make_cell(rnn_size):
  return tf.contrib.rnn.GRUCell(
      rnn_size,
      bias_initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2),
      activation=tf.nn.relu)


def encoding_layer(input_data, input_size, num_rnn_nodes,
                   num_rnn_layers):
  """Defines the encoding layer of the seq2seq model.

  Returns:
    A tuple (enc_output, enc_state) containing the output and input of the
      encoder. The output is a tensor of shape [batch_size, ?, 1], and the
      encoder state is a tuple of num_rnn_layers tensors of the shape
      [batch_size, num_rnn_nodes].
  """
  enc_cell = tf.contrib.rnn.MultiRNNCell(
      [_make_cell(num_rnn_nodes) for _ in range(num_rnn_layers)])

  enc_output, enc_state = tf.nn.dynamic_rnn(
      enc_cell, input_data,
      sequence_length=[input_size] * _get_batch_size(input_data),
      dtype=tf.float32)

  return enc_output, enc_state


def _prepend_go_tokens(output_data, go_token):
  """Concat the go tokens to the beginning of each batch of output_data.
  """
  go_tokens = tf.constant(
      go_token, shape=[_get_batch_size(output_data), 1, 1])
  return tf.concat([go_tokens, output_data], axis=1)


def decoding_layer(batch_size, num_rnn_nodes, num_rnn_layers, output_size,
                   enc_state, output_data, go_token, regularizer):
  """Defines the decoding layer of the seq2seq model.
  """
  dec_cell = tf.contrib.rnn.MultiRNNCell(
      [_make_cell(num_rnn_nodes) for _ in range(num_rnn_layers)])

  # Dense layer to translate the decoder's output at each time step.
  projection_layer = tf.layers.Dense(
      units=1,
      kernel_initializer=tf.truncated_normal_initializer(
          mean=0.0, stddev=0.1),
      kernel_regularizer=regularizer)

  # Set up a training decoder.
  training_decoder_output = None
  with tf.variable_scope("decode"):
    # During PREDICT mode, the output data is none so we can't have a training
    # model.
    if output_data is not None:
      # Prepare the target sequences we'll feed to the decoder in training
      # mode
      dec_input = _prepend_go_tokens(
          output_data, go_token)

      # Helper for the training process.
      training_helper = tf.contrib.seq2seq.TrainingHelper(
          inputs=dec_input,
          sequence_length=[output_size] * batch_size)
      training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                         training_helper,
                                                         enc_state,
                                                         projection_layer)
      training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
          training_decoder, impute_finished=True,
          maximum_iterations=output_size)

  # Set up an inference decoder.
  # Reuses the same parameters trained by the training process.
  with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
    start_tokens = tf.constant(
        go_token, shape=[batch_size, 1])

    # This is an inference helper without embedding. The sample_ids are the
    # actual output in this case (not dealing with any logits here).
    # The end_fn is always False because the data is provided by a generator
    # that will stop once it reaches output_size. This could be
    # extended to outputs of various size if we append end tokens, and have
    # the end_fn check if sample_id return True for an end token.
    inference_helper = tf.contrib.seq2seq.InferenceHelper(
        sample_fn=lambda outputs: outputs,
        sample_shape=[1],
        sample_dtype=dtypes.float32,
        start_inputs=start_tokens,
        end_fn=lambda sample_ids: False)
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        enc_state,
                                                        projection_layer)
    inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder, impute_finished=True,
        maximum_iterations=output_size)

  return training_decoder_output, inference_decoder_output


def seq2seq_model(input_data, output_data, input_size, output_size,
                  go_token, num_rnn_nodes, num_rnn_layers, regularizer):
  """Define the sequence to sequence model.

  Args:
    input_data: tensor of shape [batch_size, input_size=?, 1].
    output_data: tensor of shape [batch_size, output_size=?, 1].
    input_size: int, the number of time steps in the input data.
    output_size: int, the number of time steps in the output data.
    go_token: float, the start token that will be prepended to the output data
      during the decoder phase of the model, and signals that the output begins.
      This token must be a value that may not occur during the regular data.
    num_rnn_nodes: the number of nodes per layer in the RNN network.
    num_rnn_layers: the number of layers of the RNN network.

  Returns:
    A tuple (training_decoder_output, inference_decoder_output) of tensors of
      shapes ([batch_size, input_size, 1], [batch_size, output_size, 1]).
  """
  # Pass the input data through the encoder.
  # We'll ignore the encoder output, but use the state.
  _, enc_state = encoding_layer(input_data,
                                input_size,
                                num_rnn_nodes,
                                num_rnn_layers)

  # Pass encoder state and decoder inputs to the decoders
  training_decoder_output, inference_decoder_output = decoding_layer(
      _get_batch_size(input_data), num_rnn_nodes, num_rnn_layers, output_size,
      enc_state, output_data, go_token, regularizer)

  return training_decoder_output, inference_decoder_output


def _get_data(data, batch_size, input_size):
  """
  Get the data with the correct shape. Assumes that you pass the right value for
  the batch_size, otherwise it results in inconclusive results or reshaping
  errors.

  Args:
    data: tensor of shape [?, input_size, 1], where the first element is the
      batch_size, but may be unknown.
    batch_size: int, the target batch_size.
    input_size: the input size representing the number of time steps in data.
      Used to assert it is indeed the second element of the shape.

  """
  if data.get_shape().as_list()[1] != input_size:
    raise AssertionError('The data does size %d instead of the expected %s' %
                         (data.get_shape().as_list()[1], input_size))
  # This doesn't actually change the shape, but it makes tensorflow know that
  # the batch size is batch_size.
  # Initial tensor shape: [?, time=input_size, 1]
  # Final tensor shape: [batch_size, time=?, 1]
  reshaped = tf.reshape(data, [batch_size, -1, 1])
  return reshaped


def rnn_model_fn(features, labels, mode, params):
  """Model function."""
  print('---------- Mode:', mode.upper(), ' ----------')
  input_size = params['input_size']
  output_size = params['output_size']
  batch_size = params['batch_size']
  learning_rate = params['learning_rate']
  l2_regularization = params.get('l2_regularization')
  if l2_regularization:
    regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regularization)
  else:
    regularizer = None
  num_rnn_layers = params.get('num_rnn_layers')
  num_rnn_nodes = params.get('num_rnn_nodes')

  # Input and output data. During prediction output_data = None.
  # Input tensor shape: [batch_size, time=input_size, 1]
  # Output tensor shape: [batch_size, time=output_size, 1]
  input_data = _get_data(features['input'], batch_size, input_size)
  if mode != tf.estimator.ModeKeys.PREDICT:
    output_data = _get_data(labels['output'], batch_size, output_size)
  else:
    output_data = None
  go_token = -1.

  training_decoder_output, inference_decoder_output = seq2seq_model(
      input_data, output_data, input_size, output_size, go_token,
      num_rnn_nodes, num_rnn_layers, regularizer)

  # PREDICT Mode
  if mode == tf.estimator.ModeKeys.PREDICT:
    # Predictions tensor shape: [batch_size, output_size, 1]
    predictions = inference_decoder_output.rnn_output
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  # TRAIN or EVAL mode
  # Calculate Loss (for both TRAIN and EVAL modes)
  # Predictions tensor shape: [batch_size, output_size, 1]
  predictions = training_decoder_output.rnn_output
  loss = tf.losses.mean_squared_error(
      labels=output_data, predictions=predictions)

  # TRAIN mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate)
    # Apply gradient clipper.
    gradients = optimizer.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var
                        in gradients if grad is not None]
    train_op = optimizer.apply_gradients(
        capped_gradients, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op)

  # EVAL mode
  assert mode == tf.estimator.ModeKeys.EVAL

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss)


def train_and_evaluate_model(args):
  """Train and evaluate the model."""
  if args.train_steps % args.batch_size:
    raise ValueError(
        'The number of train steps %d must be a multiple of batch size %d.' %
        (args.train_steps, args.batch_size))

  params = {'input_size': args.time_steps - args.output_size,
            'output_size': args.output_size,
            'batch_size': args.batch_size,
            'l2_regularization': args.l2,
            'learning_rate': args.learning_rate,
            'num_rnn_nodes': args.num_rnn_nodes,
            'num_rnn_layers': args.num_rnn_layers,
            }

  # Define the estimator.
  log_step_count_steps = max(1, args.train_steps / args.batch_size /
                             args.evaluations // args.logs_per_training)
  estimator = tf.estimator.Estimator(
      model_fn=rnn_model_fn,
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
  train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: gd.input_fn(
          time_steps=args.time_steps,
          input_size=args.time_steps - args.output_size,
          batch_size=args.batch_size,
          steps=args.train_steps // args.evaluations),
      max_steps=args.train_steps // args.batch_size)

  eval_spec = tf.estimator.EvalSpec(
      input_fn=lambda: gd.input_fn(
          time_steps=args.time_steps,
          input_size=args.time_steps - args.output_size,
          batch_size=args.batch_size,
          steps=args.batch_size * args.eval_steps),
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
  test_size = max(args.batch_size, 2)
  # The models assumes a fixed batch size, so we need to provide one or we get
  # incorrect results.
  test_params = (test_params * ((args.batch_size + 1) // 2))[:test_size]
  predict_results = list(estimator.predict(
      input_fn=lambda: gd.input_fn(
          time_steps=args.time_steps,
          input_size=args.time_steps - args.output_size,
          batch_size=args.batch_size,
          steps=test_size,
          default_params=test_params)))

  test_dataset = gd.input_fn(time_steps=args.time_steps,
                             input_size=args.time_steps - args.output_size,
                             batch_size=1,
                             steps=len(test_params),
                             default_params=test_params)
  test_iterator = test_dataset.make_initializable_iterator()
  test_elem = test_iterator.get_next()

  print('Arguments:', args)
  with tf.Session() as sess:
    sess.run(test_iterator.initializer)
    for prediction in predict_results[:2]:
      test_data = sess.run(test_elem)
      print('Expected:')
      print({'y0': test_data[0]['y0'], 'v0y': test_data[0]['v0y']})
      input = np.transpose(np.asarray(test_data[0]['input']),
                           axes=[0, 2, 1])[0, 0]
      print(input)
      output = np.transpose(np.asarray(test_data[1]['output']),
                            axes=[0, 2, 1])[0, 0]
      print(output)
      print('Predicted:')
      prediction = np.transpose(np.asarray(prediction))
      print(prediction)
      print('mse:', _mse(output, prediction[0]))
      print()


def main(argv):
  args = parser.parse_args(argv[1:])
  print('Arguments:', args)

  estimator = train_and_evaluate_model(args)
  make_predictions(args, estimator)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
