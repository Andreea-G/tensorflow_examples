"""Data generation for the CNN model. Each data point is a numpy array 
representing an image.
"""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['savefig.pad_inches'] = 0

import matplotlib.pyplot as plt
import numpy as np
import random
from skimage import color
import tensorflow as tf

np.set_printoptions(threshold=np.nan, linewidth=1000, precision=2)


_GRAVITY = 10


class DataGenerator(object):

  def get_data_generator(self, steps, image_size, input_size,
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

  def __init__(self, steps, image_size):
    """Initialize.

    Args:
      steps: int, the total number of steps to perform. The output will contain
        steps/batch_size batches of size batch_size each.
      image_size: int, the combined number of time steps of the input and
        output.
    """
    self.steps = steps
    self.image_size = image_size

  def _full_frame(self, dpi, figsize, frame=None):
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    fig = plt.figure(dpi=dpi, figsize=figsize)
    ax = plt.axes(frame, frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    return fig

  def _get_bounces(self, y0, v0y):
    """Calculates the time intervals until the ball bounces off the ground, and
    its velocity on the y-axis at that moment.
    """
    vy_bounce = np.sqrt(v0y ** 2 + 2 * _GRAVITY * y0)
    t_first_bounce = (v0y + vy_bounce) / _GRAVITY
    t_next_bounces = 2 * vy_bounce / _GRAVITY
    return {'t0': t_first_bounce, 'tn': t_next_bounces, 'v0y': vy_bounce}

  def _x_of_t(self, t, v0x):
    """This is the function x(t) for a given time t.
    """
    return v0x * t

  def _y_of_t(self, t, y0, v0y, bounce=None):
    """This is the function y(t) for a given time t.
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

  def _get_image(self, y0=0, v0x=1.5, v0y=0, xmax=10, ymax=4, lw=0.1):
    """Returns a 2D matrix of size image_size containing the image.

    Args:
      y0: float, the initial height at time t = 0.
      v0x: float, the initial velocity in the x-direction. This of course
        remains constant.
      v0y: float, the initial velocity in the y-direction at t = 0.
      xmax: float, the image x-axis is from 0 to xmax.
      ymax: float, the image y-axis is from 0 to ymax.
      lw: float, the linewidth when creating the plot.

    Returns:
      A gray-scale 2D matrix.
    """
    # Get the x(t) and y(t) of size image_size for timesteps t.
    tmax = xmax / v0x
    time = np.linspace(0, tmax, self.image_size, endpoint=False)
    bounce = self._get_bounces(y0, v0y)
    x = [self._x_of_t(t, v0x) for t in time]
    y = [self._y_of_t(t, y0, v0y, bounce) for t in time]

    # Plot the image without any frames or empty borders around it.
    frame = [0, 0, 1, 1]
    fig = self._full_frame(dpi=self.image_size, figsize=(1, 1), frame=frame)
    plt.xlim([0, xmax])
    plt.ylim([0, ymax])
    plt.plot(x, y, color='k', linewidth=lw)
    fig.canvas.draw()
    plt.close()

    # Turn it into a matrix representing the gray scale pixels.
    width, height = fig.get_size_inches() * fig.get_dpi()
    mpl_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    mpl_image = mpl_image.reshape(int(height), int(width), 3)
    gray_image = 1 - color.rgb2gray(mpl_image)

    # Scale the gray scale values so they are exactly between 0 and ~0.99. Makes
    # the printed image easy to see.
    scale = gray_image.max() * 1.01
    if scale > 0:
      gray_image /= scale
    return gray_image

  def _keep_image(self, image):
    non_zero_elem = len([x for line in image for x in line if x > 0])
    return non_zero_elem / (self.image_size ** 2) > 0.01

  def get_data_generator(self, default_params=None, xmax=10, ymax=4,
                         **input_kwargs):
    """Returns the generator."""
    def get_default(i):
      if default_params and i < len(default_params):
        return default_params[i]
      return None

    def data_generator():
      for i in range(self.steps):
        while True:
          default = get_default(i)
          if default is not None:
            y0 = default.get('y0')
            v0y = default.get('v0y')
          else:
            y0 = random.uniform(0, ymax)
            vmax = np.sqrt(2 * _GRAVITY * (ymax - y0))
            v0y = random.uniform(-vmax, vmax)
          image = self._get_image(y0=y0, v0y=v0y,
                                  xmax=xmax, ymax=ymax, **input_kwargs)
          if self._keep_image(image):
            yield ({'image': image},
                   {'y0': y0, 'v0y': v0y})
            break
    return data_generator

  def get_types(self):
    """Gets the types for the Dataset.from_generator() call."""
    return ({'image': tf.float32},
            {'y0': tf.float32, 'v0y': tf.float32})

  def get_shapes(self):
    """Gets the shapes for the Dataset.from_generator() call."""
    return ({'image': (self.image_size, self.image_size)},
            {'y0': (), 'v0y': ()})


def input_fn(image_size, batch_size=1, steps=1,
             default_params=None, generator_class=None, **input_kwargs):
  """Input function to be passed to the estimators, used for training, 
  evaluation, and prediction.  

  Args:
    image_size: int, size of the square image in pixels.
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
  if generator_class is None:
    generator_class = BouncingBallsDataGenerator
  generator = generator_class(steps, image_size)
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
      [{'y0': 1, 'v0y': 7},
       {'y0': 3, 'v0y': -2},
       {'y0': 4, 'v0y': 0}]
  )
  batch_size = 1
  input_kwargs = {'lw': 0.1}
  steps = len(default_params)
  dataset = input_fn(image_size=28, batch_size=batch_size, steps=steps,
                     default_params=default_params, **input_kwargs)
  el = dataset.make_one_shot_iterator().get_next()
  with tf.Session() as sess:
    for i in range(steps // batch_size):
      print('Session', i)
      result = sess.run(el)
      for image, y0, v0y in zip(result[0]['image'],
                                result[1]['y0'],
                                result[1]['v0y']):
        print(image)
        print({'y0': y0, 'v0y': v0y})


if __name__ == "__main__":
  main()
