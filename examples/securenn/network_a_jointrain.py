# pylint:  disable=redefined-outer-name
"""An example of performing secure training with MNIST.

Reproduces Network A from SecureNN, Wagh et al.
"""
from __future__ import absolute_import
import sys
from typing import List
import time
import tensorflow as tf
import tensorflow.keras as keras
import tf_encrypted as tfe
import tf_encrypted.keras.optimizers as optimizers

from conv_convert import get_data_from_tfrecord

# tfe.set_tfe_events_flag(True)


if len(sys.argv) >= 2:
  # config file was specified
  config_file = sys.argv[1]
  config = tfe.RemoteConfig.load(config_file)
else:
  # default to using local config
  config = tfe.LocalConfig([
      'server0',
      'server1',
      'crypto-producer',
      'model-trainer',
      'prediction-client'
  ])
tfe.set_config(config)
players = ['server0', 'server1', 'crypto-producer']
prot = tfe.protocol.SecureNN(*tfe.get_config().get_players(players))
tfe.set_protocol(prot)
session_target = sys.argv[2] if len(sys.argv) > 2 else None



class Client():
  """Contains methods meant to be executed by a prediction client.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the data owner
    build_update_step: `Callable`, the function used to construct
                       a local federated learning update.
  """

  #BATCH_SIZE = 20
  BATCH_SIZE = 256
  ITERATIONS = 60000 // BATCH_SIZE
  EPOCHS = 3
  LEARNING_RATE = 3e-3
  IN_N = 28 * 28
  HIDDEN_N = 128
  OUT_N = 10

  def provide_input(self) -> List[tf.Tensor]:
    """Prepare input data for prediction."""
    with tf.name_scope('loading'):
      prediction_input, expected_result = get_data_from_tfrecord(
          "./data/test.tfrecord", self.BATCH_SIZE, flattened=True).get_next()

    with tf.name_scope('pre-processing'):
      prediction_input = tf.reshape(
          #prediction_input, shape=(self.BATCH_SIZE, ModelTrainer.IN_N))
          prediction_input, shape = (self.BATCH_SIZE, self.IN_N))
      expected_result = tf.reshape(
          expected_result, shape=(self.BATCH_SIZE,))
      expected_result = tf.one_hot(expected_result, depth=10)

    return [prediction_input, expected_result]

  def receive_output(self, likelihoods: tf.Tensor, y_true: tf.Tensor):
    with tf.name_scope('post-processing'):
      prediction = tf.argmax(likelihoods, axis=1)
      y_true=tf.argmax(y_true, axis=1)
      eq_values = tf.equal(prediction, tf.cast(y_true, tf.int64))
      acc = tf.reduce_mean(tf.cast(eq_values, tf.float32))
      op = tf.print('Expected:', y_true, '\nActual:',
                    prediction, '\nAccuracy:', acc)

      return op


if __name__ == '__main__':



  client = Client()



  x_train, y_train = tfe.define_private_input(
      'train-client', client.provide_input)  # pylint: disable=E0632
  print("x=",x_train)
  print("y=", y_train)


  x_test, y_test = tfe.define_private_input(
      'prediction-client', client.provide_input)  # pylint: disable=E0632
  print("x=",x_test)
  print("y=", y_test)

  with tfe.protocol.SecureNN():
    batch_size = 32
    #flat_dim = ModelTrainer.IN_N
    flat_dim =client.IN_N
    batch_input_shape = [batch_size, flat_dim]
    # compute prediction
    model = tfe.keras.Sequential()
    model.add(tfe.keras.layers.Dense(client.HIDDEN_N,
                                     batch_input_shape=batch_input_shape))

   # model.add(tfe.keras.layers.Activation('relu'))
    model.add(tfe.keras.layers.Dense(client.HIDDEN_N))
    model.add(tfe.keras.layers.Activation('relu'))
    model.add(tfe.keras.layers.Dense(client.OUT_N))

    model.add(tfe.keras.layers.Activation('sigmoid'))
    model.compile(optimizer=optimizers.SGD(lr=0.01) ,loss=tfe.keras.losses.MeanSquaredError())
    model.fit(x_train,y_train, epochs=20)



    #
    # send prediction output back to client
    logits = model(x_test)
    prediction_op = tfe.define_output(
        'prediction-client', [logits, y_test], client.receive_output)

    sess = tfe.Session(target=session_target)

    start_time = time.time()


    sess.run(tf.global_variables_initializer(), tag='init')

    # print("Training")
    # sess.run(cache_updater, tag='training')
    #
    # print("Set trained weights")
    # model.set_weights(params, sess)

    for _ in range(5):
      print("Predicting")
      sess.run(prediction_op, tag='prediction')

    sess.close()

    print("running time=", time.time()-start_time)
