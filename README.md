# Tensorflow Examples: CNN for Regression and Seq2seq RNN for Time-Series Data Without Embedding.

This contains examples of building neural networks using Tensorflow's latest `Estimators` API. There are two examples included here:
1. A many-to-many RNN, namely a sequence to sequence model without embedding, where the values represent real-valued time-series data. 
1. A CNN that takes images and predicts some real-valued numbers, so this is a regression, not a classification problem. 

The reason is that there are plenty of examples online for classification problems, but hardly any for real-valued outputs, so this is what we set out to do here. 

In both cases the data is randomly generated and fed through a `tf.data.Dataset` using the `from_generator()` call. I found that generating my own data instead of just reading it from some external database makes it a lot easier to understand the underlying structure and resulting tensor shape of that data.

We use the `Estimators` API and the `tf.estimator.train_and_evaluate(...)` method to train and evaluate the model. We want to train for some number of steps, then evaluate, then train again for some number of steps, then evaluate again etc. At the end we predict the output for some test data.


## Seq2seq Time Series

In this model, we're building a many-to-many RNN where we receive a time-series data from `t_0` to `t_(i-1)`, and we want to predict the values from `t_i` to `t_N` (where `i` is the `input_size` and `N - i` is the `output_size`). The time-series is describing some function `y(t)` for some family of functions, parametrized by some parameters. A basic example of such a function could be a `sine` wave parametrized by its amplitude and period. So we're dealing with real-valued inputs and outputs, which means no embedding. Since most examples online and most tutorials describe the embedding case and use `Tensorflow`'s `GreedyEmbeddingHelper` as a result, this example if far enough off the beaten path that it forced me to dig deeper into the actual implementation of `Tensorflow`.

This [tutorial][seq2seq_tutorial] was the best one I could find on sequence-to-sequence models in `Tensorflow`, although it uses `embedding` so in some places it doesn't apply to our model. I found it does an excellent job explaining all the steps in detail, but I found its major drawback was that it's not using the `estimators` API. We do use estimators in this code.

### Model Setup

- We get the data, in the form of a `Tensor` of `shape = (batch_size, time_steps, dim)` and `dtype=float32`. The input data has `input_size` time steps, and the output data has `output_size` time steps, but this could be easily extended to data of varying length. In our code `dim = 1`, but could be extended to higher dimensions as well.
- We set up an encoding layer, which takes the time series data and passes it directly through a `tf.contrib.rnn.MultiRNNCell`. We call `tf.nn.dynamic_rnn` to get the `encoding_state`. We don't care about the `encoding_output` in this case because we only start predicting the output starting at `t_i`, when the encoding stage is over.
```  
def _make_cell(num_rnn_nodes):
  return tf.contrib.rnn.GRUCell(
      rnn_size,
      bias_initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2),
      activation=tf.nn.relu)

enc_cell = tf.contrib.rnn.MultiRNNCell(
    [_make_cell(num_rnn_nodes) for _ in range(num_rnn_layers)])

enc_output, enc_state = tf.nn.dynamic_rnn(
    enc_cell, input_data,
    sequence_length=[input_size] * batch_size,
    dtype=tf.float32)
```
- We create the decoding layer, which takes the encoding_state `enc_state`. Here we have two modes of operation:
 
1. The training mode where we use `tf.contrib.seq2seq.TrainingHelper` as usual. As `inputs`, the `TrainingHelper` takes in the output data (the time series from `t_i` to `t_N`), but we need to prepend a `GO` token at the beginning. This has to be some value that will not be found in the rest of the data. E.g. if we know that the time-series values are always positive, we can just set the `GO` token to be `-1`. If we have data of varying lengths we also have to append an `END` token, otherwise we don't. Then we call `tf.contrib.seq2seq.BasicDecoder` using the above `TrainingHelper`, and obtain the `training_decoder_output`. The values of `training_decoder_output.rnn_output` represent directly the time series data.
 ```
# Prepend the GO tokens to the target sequences, which we'll feed to the 
# decoder in training mode.
dec_input = _prepend_go_tokens(output_data, go_token)

# Helper for the training process.
training_helper = tf.contrib.seq2seq.TrainingHelper(
    inputs=dec_input,
    sequence_length=[output_size] * batch_size)
training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                   training_helper,
                                                   encoding_state,
                                                   projection_layer)
training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
    training_decoder, impute_finished=True,
    maximum_iterations=output_size)
```
The `projection_layer` is a `tf.layers.Dense` layer whose `units` much match the dimension `dim` of the output. This will translate the decoder's output at each time step and return the output time series.

2. The evaluation mode is where things are most different for our problem, since we can no longer use the `tf.contrib.seq2seq.GreedyEmbeddingHelper`. Instead, it is very simple to use a `tf.contrib.seq2seq.InferenceHelper` like following:
```
inference_helper = tf.contrib.seq2seq.InferenceHelper(
    sample_fn=lambda outputs: outputs,
    sample_shape=[dim],
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
```
In this case, the `sample_fn` just returns the unmodified `outputs`. The `start_tokens` are a `Tensor` of `shape = (batch_size, dim)` with the values equal to the `GO` token. And the `sample_ids` are just the `outputs`, so `end_fn` is a function that takes the `outputs` and returns `True` if that is an `END` token. If we're dealing with data of a fixed length then we don't actually need `END` tokens.

- We use the `training_decoder_output` and `inference_decoder_output` to define the `tf.estimator.EstimatorSpec` for the respective `train`, `eval` and `predict` modes. We calculate the loss as the `mean_squared_error` between the output data (our output time series) and the `rnn_output` of the decoder outputs. 
- We create the `tf.estimator.Estimator` using the `EstimatorSpec` above, and we train and evaluate. This is also something that took some digging to figure out: we want to train for some time, then evaluate, then train again, then evaluate etc. `train_steps` is the total number of steps we want to train for, which is processed in batches of `batch_size`. We have `evaluations` as the total number of evaluation calls performed during the `train_and_evaluate` call. This is achieved by making the `input_fn` only `yield` data for a total of `steps=train_steps / args.evaluations` before it's done. Once the `input_fn` stops producing data after `steps` steps, the estimator switches from one mode to the other.
```
estimator = tf.estimator.Estimator(
    model_fn=rnn_model_fn,
    params=params,
    config=tf.estimator.RunConfig(
        log_step_count_steps=log_step_count_steps
    )
)
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
```
- Finally, we may call `estimator.predict(input_fn=...)` to predict the output time series for a given test input. 


## CNN

Here we're building a CNN model that receives images in batches, where each image represents a plot of some family of functions for some given parameters. A basic example of such a function could be a `sine` wave parametrized by its amplitude and period. The parameters that were used to generate the function (so the amplitude and period in this case) are real numbers, and they are the labels that the CNN has to predict. I know what you're thinking, this is probably the least efficient way of predicting the parameters describing my functions :smile:. But this is just a toy model meant to help understand `Tensorflow`, so don't read too much into it.

The plots are created using `matplotlib` and then converted into 2-D matrices (`numpy` arrays) of `image_size x image_size` pixels. We have gray scale images so one color channel, and each point in the matrix is a value between `0` and `1`.

I used the [Tensorflow's CNN Tutorial for MNIST data][cnn_mnist_tutorial] as a starting point, but we now have real-valued outputs instead of a classification problem. In addition, just like for the RNN model above, the data is randomly generated, which also means we have full control over our data to play with the `image_size`, the type of function, etc. And once again, this code shows how we can train and evaluate, which is not explained in the official tutorial (train for some number of steps, then evaluate, then train again etc). At the end predict the labels for some user-provided test images.

### Model Setup

- We get the data in batches of images, which we shape in the form of a `Tensor` of `shape = (batch_size, width, height, channels)` and `dtype=float32`. In our chase `channels = 1` because we're using gray scale images.  
- We apply a couple of convolution + max pooling layers.
- We flatten the final pooling layer, add a dense layer and a dropout layer. So far it's pretty similar to the [MNIST tutorial][cnn_mnist_tutorial].
- At this stage though, we apply a dense layer with the `units` equal to the number of labels that we're trying to predict. This is our `output_layer`:
```
output_layer = tf.layers.dense(inputs=dropout, units=2,
                               kernel_regularizer=regularizer)
```
- Get the `tf.estimator.EstimatorSpec` for the train, eval and predict modes respectively. The `loss` is the `mean_squared_error` for the output layer.
- Create the `tf.estimator.Estimator` from the `tf.estimator.EstimatorSpec`, and then train and evaluate just like for the RNN model above.
- Finally generate some custom test images from some test parameters, and predict those parameters by calling `estimator.predict(input_fn=...)`.

## Data Generators

The data generation is pretty similar for our two models. In both cases we generate some parametrized functions for randomly generated parameters. For the RNN model we return a vector containing the function values at discrete time steps. For the CNN model we plot the function in gray scale mode and convert the plot into a 2-D matrix, while cutting out any axes or labels. 
We then use the `tf.data.Dataset.from_generator(...)` call to create a dataset:
```
def input_fn(generator_class, batch_size=1, steps=1, ....):
  generator = generator_class(steps, ...)
  dataset = tf.data.Dataset.from_generator(
      generator.get_data_generator(...),
      output_types=generator.get_types(),
      output_shapes=generator.get_shapes()
  )
  dataset = dataset.batch(batch_size=batch_size)
  return dataset
```
The `generator_class` is a custom class we created, where the `get_data_generator(...)` method returns a callable. This callable yields data for a total of `steps` steps before it stops. The data yielded by the callable is a tuple `(features, labels)`, where the `features` and `labels` are dictionaries.

## Data Functions Used In This Code (Optional)

This part is not in the slightest needed to understanding `Tensorflow`, but I should probably explain the functions I'm using here. I figured using `sin` or `cos` was too boring, so decided to come up with something more intersting. Picture yourself playing basketball, and you're throwing the ball in the air. The ball will move according to a parabolic trajectory, then it will hit the ground and bounce back again. Unlike in the real world, here we assume zero energy loss. So in our very unrealistic scenario the ball would continue forever to create these parabolic trajectories, bouncing off the ground over and over again. The parameters that I'm randomizing are the height `y0` from which you throw the ball, and the initial velocity in the vertical direction `v0y`. The height will always be a positive number (the ground level is at `y = 0`), but `v0y` can be positive or negative (throw the ball up or down). The other variables I'm keeping constant (the max time, the horizontal velocity `v0x`, the gravitational constant etc), those just set the scale of the image. 

## Licence

GNU GPL version 2 or any later version


[cnn_mnist_tutorial]: https://www.tensorflow.org/tutorials/estimators/cnn
[seq2seq_tutorial]: https://github.com/udacity/deep-learning/blob/master/seq2seq/sequence_to_sequence_implementation.ipynb

