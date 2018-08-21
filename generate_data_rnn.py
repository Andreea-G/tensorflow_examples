"""Data generation for the RNN model. Each data point is a 1D time-series."""

import numpy as np
import random
import tensorflow as tf

np.set_printoptions(threshold=np.nan, linewidth=1000, precision=2)

_GRAVITY = 10


class DataGenerator(object):

  def get_data_generator(self, steps, time_steps, input_size,
                         label_defauls=None, **input_kwargs):
    """Returns a callable generator that yields a tuple of the form
    (input, output).
    """
    raise NotImplementedError("get_data has not been implemented")

  def get_types(self):
    """Gets the types for the Dataset.from_generator() call."""
    raise NotImplementedError("get_types has not been implemented")

  def get_shapes(self):
    """Gets the shapes for the Dataset.from_generator() call."""
    raise NotImplementedError("get_shapes has not been implemented")


class BouncingBallsDataGenerator(DataGenerator):
  """A DataGenerator that describes the trajectory for a bouncing ball, for
  given initial conditions, assuming no loss of energy. This then results in a
  series of upside-down parabolas. The resulting data is a vector of heights
  'y' for each time step.
  """

  def __init__(self, steps, time_steps, input_size):
    """Initialize.

    Args:
      steps: int, the total number of steps to perform. The output will contain
        steps/batch_size batches of size batch_size each.
      time_steps: int, the combined number of time steps of the input and
        output.
      input_size: int, the input size. If None, it is equal to time_steps, and
        no output is provided.
    """
    if input_size > time_steps:
      raise ValueError(
          "input_size %d must not be larger than time_steps %d" %
          (input_size, time_steps))
    self.steps = steps
    self.time_steps = time_steps
    self.input_size = input_size

  def _get_bounces(self, y0, v0y):
    """Calculates the time intervals until the ball bounces off the ground, and
    its velocity on the y-axis at that moment.
    """
    vy_bounce = np.sqrt(v0y ** 2 + 2 * _GRAVITY * y0)
    t_first_bounce = (v0y + vy_bounce) / _GRAVITY
    t_next_bounces = 2 * vy_bounce / _GRAVITY
    return {'t0': t_first_bounce, 'tn': t_next_bounces, 'v0y': vy_bounce}

  def _y_of_t(self, t, y0, v0y, bounce=None):
    """This is the function y(t) for a given time step.
    """
    if bounce is not None:
      if bounce['v0y'] == 0:
        return 0
      if t > bounce['t0']:
        y0 = 0
        v0y = bounce['v0y']
        t -= bounce['t0']
        while t > bounce['tn']:
          t -= bounce['tn']
    return y0 + v0y * t - _GRAVITY * t ** 2 / 2

  def _get_y_values(self, y0=0, v0x=1.5, v0y=0, xmax=10):
    """Returns a vector of y values, one value per time step.

    Args:
      y0: float, the initial height at time t = 0.
      v0x: float, the initial velocity in the x-direction. This of course
        remains constant.
      v0y: float, the initial velocity in the y-direction at t = 0.
      xmax: the position of the ball at the end.

    Returns:
      A vector of y values.
    """
    tmax = xmax / v0x
    time = np.linspace(0, tmax, self.time_steps, endpoint=False)
    bounce = self._get_bounces(y0, v0y)
    return [self._y_of_t(t, y0, v0y, bounce) for t in time]

  def get_data_generator(self, default_params=None, xmax=10, ymax=4,
                         **input_kwargs):
    """Returns the generator."""
    def get_default(i):
      if default_params and i < len(default_params):
        return default_params[i]
      return None

    def data_generator():
      for i in range(self.steps):
        default = get_default(i)
        if default is not None:
          y0 = default.get('y0')
          v0y = default.get('v0y')
        else:
          y0 = random.uniform(0, ymax)
          vmax = np.sqrt(2 * _GRAVITY * (ymax - y0))
          v0y = random.uniform(-vmax, vmax)
        y = self._get_y_values(y0=y0, v0y=v0y, xmax=xmax, **input_kwargs)
        # Transpose.
        y = [[i] for i in y]
        yield ({'input': y[:self.input_size], 'y0': y0, 'v0y': v0y},
               {'output': y[self.input_size:]})
    return data_generator

  def get_types(self):
    """Gets the types for the Dataset.from_generator() call."""
    return (
        {'input': tf.float32, 'y0': tf.float32, 'v0y': tf.float32},
        {'output': tf.float32})

  def get_shapes(self):
    """Gets the shapes for the Dataset.from_generator() call."""
    return (
        {'input': (self.input_size, 1), 'y0': (), 'v0y': ()},
        {'output': (self.time_steps - self.input_size, 1)})


def input_fn(time_steps, input_size=None, batch_size=1, steps=1,
             default_params=None, generator_class=None, **input_kwargs):
  """Input function to be passed to the estimators, used for training, 
  evaluation, and prediction.  

  Args:
    time_steps: int, the combined number of time steps of the input and output.
    input_size: int, the input size. If None, it is equal to time_steps, and no
      output is provided.
    batch_size: int, the training and evaluation batch size.
    steps: int, the total number of steps to perform. The output will contain 
      steps/batch_size batches of size batch_size each.
    default_params: list[dict], a list of dictionaries used during the
      prediction method. If you want to get input/output data
      corresponding to certain parameters, then pass those parameters here. If
      None, the data generated is random. If steps > len(default_params), then
      the steps after len(default_params) are also random.
    generator_class: DataGenerator, a class that extends DataGenerator.
    **input_kwargs: additional kwargs passed to generator_class's method
      get_data_generator.

  Returns:
    A 'tf.data.Dataset' object. Outputs of `Dataset` object must be a tuple
    (features, labels).
  """
  if input_size is None:
    input_size = time_steps
  if generator_class is None:
    generator_class = BouncingBallsDataGenerator
  generator = generator_class(steps, time_steps, input_size)
  dataset = tf.data.Dataset.from_generator(
      generator.get_data_generator(default_params=default_params,
                                   **input_kwargs),
      output_types=generator.get_types(),
      output_shapes=generator.get_shapes()
  )
  dataset = dataset.batch(batch_size=batch_size)
  return dataset


def main():
  default_params = (
      [{'y0': 1, 'v0y': 7}, {'y0': 3, 'v0y': -2}, {'y0': 4, 'v0y': 0}] * 3
  )
  batch_size = 3
  steps = len(default_params)
  dataset = input_fn(time_steps=32, input_size=28, batch_size=batch_size,
                     steps=len(default_params),
                     default_params=default_params)
  el = dataset.make_one_shot_iterator().get_next()
  with tf.Session() as sess:
    for i in range(steps // batch_size):
      res = sess.run(el)
      print('Session', i)
      input = np.transpose(np.asarray(res[0]['input']), axes=[0, 2, 1])
      print(input)
      output = np.transpose(np.asarray(res[1]['output']), axes=[0, 2, 1])
      print(output)
      print({'y0': res[0]['y0'], 'v0y': res[0]['v0y']})


if __name__ == "__main__":
  main()
