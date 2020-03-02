import tensorflow as tf
import numpy as np
export_dir='./save_model'

w=tf.Variable(initial_value=tf.zeros([3,1], dtype='float32'))

b=tf.Variable(initial_value=tf.zeros([1],dtype='float32'))

x=tf.placeholder('float32', shape=[None ,3])

predictions=tf.matmul(x,w)+b

print(predictions)

y=tf.placeholder('float32', shape=[None, 1])

loss=tf.losses.mean_squared_error(y, predictions)

optimizer = tf.train.GradientDescentOptimizer(0.1)

train_op= optimizer.minimize(loss)


tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
tensor_info_y = tf.saved_model.utils.build_tensor_info(predictions)




signature = tf.saved_model.signature_def_utils.build_signature_def(
    inputs={'x': tensor_info_x},
    outputs={'y': tensor_info_y},
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)





def savedmodel(sess, signature, export_dir):
    # export_dir = os.path.join(path, str(FLAGS.model_version))
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_y':
                signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature
        })
    builder.save()



featues=np.random.normal(size=[100,3])

label=np.matmul(featues, [[1], [2], [3]])+[4] #+ np.random.normal(size=[100,1])

print(label)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for step in range(100):
        _loss, _w,  _ =sess.run([loss, w, train_op],feed_dict={x:featues, y: label})

        if step%10==0:
            print("step=", step)
            print("loss=", _loss)
            print("w=", _w)

    #savedmodel(sess, signature, export_dir)