#! /usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import math

print(tf.__version__)


def gen_data(w, b, num_points):
    '''
    y=wx+b
    :param w:
    :param b:
    :param num_points:
    :return:
    '''
    vectors_set = []
    x_data = []
    y_data = []
    for i in range(num_points):
        x1 = np.random.normal(0.0, 0.55)  # 横坐标，进行随机高斯处理化，以0为均值，以0.55为标准差
        y1 = x1 * w + b + np.random.normal(0.0, 0.03)  # 纵坐标，数据点在y1=x1*0.1+0.3上小范围浮动
        vectors_set.append([x1, y1])
        x_data = [v[0] for v in vectors_set]
        y_data = [v[1] for v in vectors_set]
        #plt.scatter(x_data, y_data, c='r')
    #plt.show()
    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)
    return x_data, y_data


def square_loss(array1, array2):
    '''
    使用math自定义平方损失函数：Square loss=(x-y)^2
    :param array1: input x
    :param array2: input y
    :return:
    '''
    # loss=np.square(array1-array2)
    square = []
    for a1, a2 in zip(array1, array2):
        s = math.pow(a1 - a2, 2)
        square.append(s)
    loss = np.array(square, dtype=np.float32)
    return loss


@tf.RegisterGradient("LossGradient")
def square_loss_grad(op, grad):
    '''
      使用修饰器，建立梯度反向传播函数。其中op.input包含输入值、输出值，grad包含上层传来的梯度
      :param op:
      :param grad:
      :return:
      '''
    x = op.inputs[0]
    y = op.inputs[1]
    # 计算平方损失的梯度：loss=(x-y)^2
    grad_x = 2 * grad * (x - y)  # 对x求导：grad_x=2(x-y)
    grad_y = tf.negative(2 * grad * (x - y))  # 对y求导：grad_y=-2(x-y)
    return grad_x, grad_y


def my_loss(y, y_data):
    with tf.get_default_graph().gradient_override_map({"PyFunc": 'LossGradient'}):
        loss = tf.py_func(square_loss, inp=[y, y_data], Tout=tf.float32, name="PyFunc1")
        print("PyFunc1:", tf.get_default_graph().get_operation_by_name('PyFunc1'))
       # ('PyFunc1:', < tf.Operation 'PyFunc1' type=PyFunc >)
    return loss


def train_linear_regression(x_data, y_data, max_iterate):
    '''
    :param x_data:
    :param y_data:
    :param max_iterate: 最大迭代次数
    :return:
    '''
    print("x_data.shape:{}".format(x_data.shape))
    print("y_data.shape:{}:".format(y_data.shape))

    # 定义线性回归模型
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')  # 生成1维的W矩阵，取值是[-1,1]之间的随机数
    b = tf.Variable(tf.zeros([1]), name='b')  # 生成1维的b矩阵，初始值是0
    y = W * x_data + b  # 经过计算得出预估值y

    # 定义计算图
    graph = tf.get_default_graph()


    loss=my_loss(y, y_data)
    print("loss:", loss)

    loss = tf.reduce_mean(loss)

    # 定义优化器
    optimizer = tf.train.GradientDescentOptimizer(0.1)  # 采用梯度下降法来优化参数  学习率为0.1
    train = optimizer.minimize(loss, name='train')  # 训练的过程就是最小化这个误差值

    #print(tf.get_default_graph().as_graph_def())

    # 训练
    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for step in range(max_iterate):  # 执行20次训练
            _, pre_W, pre_b, pre_loss = sess.run([train, W, b, loss])
            print("step:{},W={},b={},loss={}".format(step + 1, pre_W, pre_b, pre_loss))




if __name__ == '__main__':
    w = 0.1
    b = 0.3
    num_points = 1000
    max_iterate = 1000
    x_data, y_data = gen_data(w, b, num_points)
    train_linear_regression(x_data, y_data, max_iterate)

