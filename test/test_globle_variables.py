import tf_encrypted as tfe
import tensorflow as tf


with tfe.Session() as sess:
    sess.run(tf.global_variables_initializer(),tag='init')
sess.close()
with tfe.Session() as sess:
    sess.run(tf.global_variables_initializer(),tag='init')