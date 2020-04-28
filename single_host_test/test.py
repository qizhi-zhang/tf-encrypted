import tensorflow as tf


ps_hosts=["127.0.0.1:2222/xxx"]
worker_hosts=["127.0.0.1:2223"]


# create cluster, 创建集群信息
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
# create the server， 在当前节点上启动server，并传入集群信息，这样当前节点就可以和集群中的节点通信了
print("cluster:", cluster)
server = tf.train.Server(cluster, job_name="ps", task_index=0)
print("server.target:", server.target)