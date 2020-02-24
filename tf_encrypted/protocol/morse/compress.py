from __future__ import absolute_import

from typing import Optional, Tuple, List
import math
import sys
import tf_encrypted as tfe

import numpy as np
import tensorflow as tf
import numpy as np
from tf_encrypted.tensor.factory import AbstractFactory
from tf_encrypted.tensor.native import AbstractTensor
from tf_encrypted.tensor import int64factory, int100factory, native_factory

_thismodule = sys.modules[__name__]


def compress_int32(x: AbstractTensor)-> Tuple[AbstractFactory, tf.Tensor , tf.TensorShape]:
    """

    :param x: [n1, n2, ..., nr]
    :return:   x.factory
                tf.Tensor: [n1 x n2 x nr / compress_ratio] int32
                x.shape

    """


    modulus=x.factory.modulus


    #compress_ratio=dtype_bitsize//data_bitsize  # 一个native type可以装下几个 module type ?
    compress_ratio=int(math.floor(32 * math.log(2, modulus)) ) # 一个int64可以装下几个 module type ?


    x_flatten=tf.reshape(x.to_native(), [-1])

    x_flatten_int32=tf.cast(x_flatten, tf.int32)

    flat_dim=x_flatten_int32.shape.as_list()[0]

    result_dim= int(math.ceil(1.0*flat_dim/compress_ratio))

    pad_dim=compress_ratio*result_dim - flat_dim


    x_pad=tf.pad(x_flatten_int32, paddings=[[0,  pad_dim]])

    x_pad_reshape=tf.reshape(x_pad, [ result_dim, compress_ratio] )

    powers=np.power(modulus, range(compress_ratio)) # [0, modulus, modulus^2, ..., modulus^(compress_ratio-1)]

    powers=tf.reshape(tf.constant(powers, dtype=tf.int32), [1, compress_ratio])


    x_compress =  x_pad_reshape * powers

    x_compress = tf.reduce_sum(x_compress, axis=[1])

    return (x.factory, x_compress,  x.shape)


def de_compress_int32(factory: AbstractFactory, x_compress: tf.Tensor, original_shape: tf.TensorShape)-> AbstractTensor:

    """
    :param: factory:   x.factory
            x_compress:    tf.Tensor: [ceil(n1 x n2 x nr / compress_ratio) ] int32
            original_shape:    x.shape
     return:   x: [n1, n2, ..., nr]
    """

    modulus=factory.modulus
    compress_ratio=int(math.floor(32 * math.log(2, modulus)))  # 一个int64可以装下几个 module type ?

    powers=np.power(modulus, range(compress_ratio)) # [0, modulus, modulus^2, ..., modulus^(compress_ratio-1)]

    powers=tf.reshape(tf.constant(powers, dtype=tf.int32), [1, compress_ratio])



    x_compress=tf.expand_dims(x_compress,[-1])
    x=(x_compress//powers)%modulus

    x=tf.cast(x, factory.native_type)

    x=tf.reshape(x, [-1])

    original_dim=np.prod(original_shape.as_list())
    pad_dim=x.shape.as_list()[0]-original_dim

    x, _ =tf.split(x,[original_dim, pad_dim] )

    x=tf.reshape(x, original_shape)

    return factory.tensor(x)



def compress_int64(x: AbstractTensor)-> Tuple[AbstractFactory, tf.Tensor , tf.TensorShape]:
    """

    :param x: [n1, n2, ..., nr]
    :return:   x.factory
                tf.Tensor: [n1 x n2 x nr / compress_ratio] int64
                x.shape

    """


    modulus=x.factory.modulus


    #compress_ratio=dtype_bitsize//data_bitsize  # 一个native type可以装下几个 module type ?
    compress_ratio=int(math.floor(64 * math.log(2, modulus)) ) # 一个int64可以装下几个 module type ?


    x_flatten=tf.reshape(x.to_native(), [-1])

    x_flatten_int64=tf.cast(x_flatten, tf.int64)

    flat_dim=x_flatten_int64.shape.as_list()[0]

    result_dim= int(math.ceil(1.0*flat_dim/compress_ratio))

    pad_dim=compress_ratio*result_dim - flat_dim


    x_pad=tf.pad(x_flatten_int64, paddings=[[0,  pad_dim]])

    x_pad_reshape=tf.reshape(x_pad, [ result_dim, compress_ratio] )

    powers=np.power(modulus, range(compress_ratio)) # [0, modulus, modulus^2, ..., modulus^(compress_ratio-1)]

    powers=tf.reshape(tf.constant(powers, dtype=tf.int64), [1, compress_ratio])


    x_compress =  x_pad_reshape * powers

    x_compress = tf.reduce_sum(x_compress, axis=[1])

    return (x.factory, x_compress,  x.shape)


def de_compress_int64(factory: AbstractFactory, x_compress: tf.Tensor, original_shape: tf.TensorShape)-> AbstractTensor:

    """
    :param: factory:   x.factory
            x_compress:    tf.Tensor: [ceil(n1 x n2 x nr / compress_ratio) ] int64
            original_shape:    x.shape
     return:   x: [n1, n2, ..., nr]
    """

    modulus=factory.modulus
    compress_ratio=int(math.floor(64 * math.log(2, modulus)))  # 一个int64可以装下几个 module type ?

    powers=np.power(modulus, range(compress_ratio)) # [0, modulus, modulus^2, ..., modulus^(compress_ratio-1)]

    powers=tf.reshape(tf.constant(powers, dtype=tf.int64), [1, compress_ratio])



    x_compress=tf.expand_dims(x_compress,[-1])
    x=(x_compress//powers)%modulus

    x=tf.cast(x, factory.native_type)

    x=tf.reshape(x, [-1])

    original_dim=np.prod(original_shape.as_list())
    pad_dim=x.shape.as_list()[0]-original_dim

    x, _ =tf.split(x,[original_dim, pad_dim] )

    x=tf.reshape(x, original_shape)

    return factory.tensor(x)



def compress_fake(x: AbstractTensor)-> Tuple[AbstractFactory, tf.Tensor , tf.TensorShape]:
    compressed=x.to_native()
    return x.factory, compressed, compressed.shape

def decompress_fake(factory: AbstractFactory, x_compress: tf.Tensor, original_shape: tf.TensorShape)-> AbstractTensor:
    return factory.tensor(x_compress)


if __name__=='__main__':
    ZZ128 = native_factory(np.int32, 128)

    # x = np.random.randint(128, size=[2])
    x = np.array(range(128)).reshape(16,8)
    print("x=", x)

    x = ZZ128.tensor(x)

    factory, x_compress,  shape= compress_int64(x)

    print(factory, x_compress,  shape)

    y=de_compress_int64(factory, x_compress,  shape)

    print(y)

    with tfe.Session() as sess:
        print("x_compress:",sess.run(x_compress))
        print("y:", sess.run(y))


