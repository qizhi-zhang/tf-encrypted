# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: tiangc <tianguicheng@chuangxin.com>
# Date:   2018/11/12

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.python.client import timeline

tf.app.flags.DEFINE_string("ps_hosts", "morse_tfe_ps:2222", "ps hosts")
tf.app.flags.DEFINE_string("worker_hosts", "morse_tfe_worker:2223", "worker hosts")
tf.app.flags.DEFINE_string("job_name", "worker", "'ps' or'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main():
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    '''
    下面两行代码，对于所有节点来说是一样的
    '''
    # create cluster, 创建集群信息
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    print("cluster:", cluster)
    # create the server， 在当前节点上启动server，并传入集群信息，这样当前节点就可以和集群中的节点通信了
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    print("server.target:", server.target)
    # server.join()




    with tf.variable_scope("new2"):

        if FLAGS.job_name == 'ps':
            server.join()
            #print("ps")
        else:
            with tf.device('/job:ps/task:0/cpu:0'):
                print([n.name for n in tf.get_default_graph().as_graph_def().node])
                # input_data = tf.Variable(
                #     [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]],
                #     name="input_data")

                # input_data = tf.Variable(
                #     [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.],[1,1,1]],
                #     name="input_data")
                input_data = tf.get_variable(initializer=
                    [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.], [1, 1, 1]],
                    name="input_data")
                b = tf.Variable([[1.], [1.], [2.]], name="w")


            with tf.device("/job:worker/task:0/cpu:0"):
                output=tf.matmul(input_data, b)

            # 图内复制，只在worker0上创建client
            with tf.Session("grpc://localhost:2223") as sess:

                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                print(sess.run(output))


if __name__ == "__main__":
    main()