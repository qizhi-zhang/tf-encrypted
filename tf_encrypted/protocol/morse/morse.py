from __future__ import absolute_import

from typing import Optional, Tuple, List
import math
import random
import sys

import numpy as np
import tensorflow as tf
import numpy as np
from tf_encrypted.config import get_config
from tf_encrypted.player import Player
from tf_encrypted.protocol import memoize, nodes
from tf_encrypted.protocol.securenn.odd_tensor import oddint64_factory
from tf_encrypted.protocol.pond import Pond, PondTensor, PondPublicTensor, PondPrivateTensor, PondMaskedTensor
from tf_encrypted.protocol.securenn import  SecureNN
from tf_encrypted.tensor import native_factory, int64factory
#from tf_encrypted.tensor.factory import AbstractFactory, AbstractTensor
from tf_encrypted.tensor.factory import AbstractFactory
from tf_encrypted.tensor.native import AbstractTensor


_thismodule = sys.modules[__name__]


# def get_shift_tensor(n):
#     x=np.trim_zeros(n)
#     for i in range(n):
#         for j in range(n):
#             for k in range(n):
#                 if ((i+j)%n)==k:
#                     x[i,j,k]=1
#     return x


class Morse(SecureNN):
    """
    SecureNN(server_0, server_1, server_2, prime_factory, odd_factory, **kwargs)

    Implementation of SecureNN from
    `Wagh et al <https://eprint.iacr.org/2018/442/>`_.
    """

    def __init__(self,
               server_0: Optional[Player] = None,
               server_1: Optional[Player] = None,
               server_2: Optional[Player] = None,
               tensor_factory: Optional[AbstractFactory] = None,
               prime_factory: Optional[AbstractFactory] = None,
               odd_factory: Optional[AbstractFactory] = None,
               **kwargs) -> None:
        server_0 = server_0 or get_config().get_player('server0')
        server_1 = server_1 or get_config().get_player('server1')
        server_2 = server_2 \
            or get_config().get_player('server2') \
            or get_config().get_player('crypto-producer')

        assert server_0 is not None
        assert server_1 is not None
        assert server_2 is not None

        super(Morse, self).__init__(
            server_0=server_0,
            server_1=server_1,
            server_2=server_2,
            tensor_factory=tensor_factory,
            **kwargs)






    def assistant_OT(self, y: AbstractTensor, j: AbstractTensor)->  AbstractTensor:
        #innput: y [n1,n2,...,nk-1, nk ] in device0,  j [n1, n2, ..., nk-1] in device1 modulus=nk
        # output: yj in device1 [n1, n2, ..., nk-1] dtype(yj)==dtype(y)


        assert y.shape[-1]==j.factory.modulus

        with tf.device(self.server_2.device_name):
            x=y.factory.sample_uniform(y.shape)

            i = j.factory.sample_uniform(j.shape)
            x_reshape=x.reshape(axes=[-1, x.shape[-1]])
            i_reshape=i.reshape(axes=[-1])
            i_one_hot=tf.one_hot(indices=i_reshape.value, depth=tf.shape(x_reshape.value)[1], axis=-1, dtype=x_reshape.value.dtype)
            #i_one_hot=native_factory(np.int32, 67).tensor(i_one_hot) #x_reshape.factory
            i_one_hot=x_reshape.factory.tensor(i_one_hot)
            xi_reshape=(x_reshape*i_one_hot).reduce_sum(axis=[1])
            #xi=xi_reshape.reshape(axes=i.shape)
        with tf.device(self.server_1.device_name):
            k=i-j
            k_reshape = k.reshape(axes=[-1])


        with tf.device(self.server_0.device_name):

            x_reshape_shift=cycle_lshift(x_reshape, k_reshape)
            #print("x_reshape_shift:",x_reshape_shift)
            y_reshape=y.reshape(axes=[-1, y.shape[-1]])
            #print("y_reshape:", y_reshape)
            z_reshape=y_reshape-x_reshape_shift

        with tf.device(self.server_1.device_name):

            j_reshape=j.reshape(axes=[-1])
            j_one_hot=tf.one_hot(indices=j_reshape.to_native(), depth=z_reshape.shape[1]   ,axis=-1, dtype=j_reshape.to_native().dtype)
            j_one_hot = z_reshape.factory.tensor(j_one_hot)
            #print("z_reshape:", z_reshape)
            #print("j_one_hot:", j_one_hot)
            zj_reshape=(z_reshape*j_one_hot).reduce_sum(axis=[1])
            yj_reshape=zj_reshape+xi_reshape
            yj=yj_reshape.reshape(axes=j.shape)
            return yj
    def get_indexed_tensor(self, x: AbstractTensor, i: AbstractTensor)->  AbstractTensor:
        """

        :param x:  [n1, n2, ..., nk-1, nk] over some R
        :param i:  [n1, n2, ..., nk-1]  over Z/nkZ
        :return:  xi  [n1, n2, ..., nk-1] over R
        """
        assert x.shape[-1]==i.factory.modulus
        i_one_hot = tf.one_hot(indices=i.to_native()%(x.shape[-1]), depth=x.shape[-1], axis=-1,
                               dtype=x.to_native().dtype)
        i_one_hot = x.factory.tensor(i_one_hot)
        xi=(x*i_one_hot).reduce_sum(axis=[-1])
        return xi

    def random_assistant_OT(self, y: AbstractTensor, j: AbstractTensor)-> PondPrivateTensor:
        # innput: y [n1,n2,...,nk-1, nk ] on device0,  j [n1, n2, ..., nk-1] on device1 mod nk
        # output: yj=yjL+yjR   [n1, n2, ..., nk-1] on same type with y
        with tf.device(self.server_0.device_name):
            yjL=y.factory.sample_uniform(y.shape[0:-1])

            yR=y-yjL.expand_dims(axis=-1)

        yjR=self.assistant_OT(yR, j)

        yj=PondPrivateTensor(self, share0=yjL, share1=yjR, is_scaled=False)

        return yj


    def left_equal_right_small(self,x: AbstractTensor, y: AbstractTensor, output_modulus: int)-> PondPrivateTensor:
        """

        :param x: AbstractTensor [n1, ..., nk] on device0
        :param y: AbstractTensor [n1, ..., nk] on device1  y.shape==x.shape,  y.modulus==x.modulus
        :return:  PondPrivateTensor [n1, ..., nk] on device0,1   modulus=output_modulus
        """
        input_modulus=x.factory.modulus
        assert input_modulus==y.factory.modulus

        with tf.device(self.server_0.device_name):
            x_one_hot=tf.one_hot(indices=x.to_native(), depth=input_modulus, axis=-1,
                                   dtype=x.to_native().dtype)

            x_one_hot = native_factory(np.int32, output_modulus).tensor(x_one_hot)

        return self.random_assistant_OT(x_one_hot, y)


    def euqal_zero_small(self, x: PondPrivateTensor, output_modulus: int)->PondPrivateTensor:
        with tf.device(self.server_1.device_name):
            mxr=0-x.share1
        return self.left_equal_right_small(x.share0, mxr, output_modulus)


    def left_equal_right_bits(self, x_bits: AbstractTensor, y_bits: AbstractTensor,
                              output_modulus: int = 128) -> PondPrivateTensor:
        """

        :param x_bits:  in device 0 F2
        :param y_bits:  in device 1 F2
        :param output_modulus:  输出值的modulus
        :return:
        """

        # x in device0, y in device 1
        with tf.device(self.server_0.device_name):
            # x_bits=x.bits()
            x_bits_onehot = tf.one_hot(indices=x_bits.to_native(), depth=2, axis=-1, dtype=np.uint8)
            ZZN = native_factory(np.int32, EXPLICIT_MODULUS=output_modulus)
            x_bits_onehot = ZZN.tensor(x_bits_onehot)

        with tf.device(self.server_1.device_name):
            # y_bits=y.bits()
            ZZ2 = native_factory(np.int32, EXPLICIT_MODULUS=2)
            y_bits = ZZ2.tensor(y_bits.to_native())

        eq = self.random_assistant_OT(x_bits_onehot, y_bits)

        return eq



    def left_equal_right(self, x: AbstractTensor, y: AbstractTensor,
                         output_modulus: int = 128) -> PondPrivateTensor:
        with tf.device(self.server_0.device_name):
            x_bits = x.bits()
        with tf.device(self.server_1.device_name):
            y_bits = y.bits()
        equal_bits=self.left_equal_right_bits(x_bits, y_bits, output_modulus)

        nequal_bits=1-equal_bits

        equal=self.euqal_zero_small(nequal_bits.reduce_sum(axis=-1),output_modulus)
        return equal

    def left_leq_right_bits(self, x: AbstractTensor, y: AbstractTensor, output_modulus: int=128) -> PondPrivateTensor:
        """
        \
        :param x: in device 0   F2
        :param y: in device 1 F2
        :return:  1 if x<=y else 0 in Z/(output_modulus)Z
        """
        with tf.device(self.server_0.device_name):
            x_bits_onehot = tf.one_hot(indices=x.to_native(), depth=2, axis=-1, dtype=np.uint8)

            ones_like_x = tf.ones_like(x.to_native())
            ones_like_x_onehot = tf.one_hot(indices=ones_like_x, depth=2, axis=-1, dtype=np.uint8)

            x_bits_or_ones_onehot =x_bits_onehot+ones_like_x_onehot-x_bits_onehot*ones_like_x_onehot

            ZZN = native_factory(np.int32, EXPLICIT_MODULUS=output_modulus)
            x_bits_or_ones_onehot = ZZN.tensor(x_bits_or_ones_onehot)

        with tf.device(self.server_1.device_name):

            ZZ2 = native_factory(np.int32, EXPLICIT_MODULUS=2)
            y = ZZ2.tensor(y.to_native())

        #print("x_bits_plus_ones_onehot:", x_bits_plus_ones_onehot)
        #print("y:", y)
        left_leq_right=self.random_assistant_OT(x_bits_or_ones_onehot, y)
        return left_leq_right

    def left_leq_right(self,x: AbstractTensor, y: AbstractTensor, output_modulus: int=128) -> PondPrivateTensor:
        """

        :param x: in device 0
        :param y: in device 1
        :return:  1 if x<=y else 0 in PondPrivateTensor
        """
        with tf.device(self.server_0.device_name):
            x_bits=x.bits()
            #x_bits_expand=tf.expand_dims(x_bits.to_native,axis=-1)
            ones=tf.ones_like(x_bits.to_native())


            ones_plusdim=tf.expand_dims(ones,-1)*tf.expand_dims(ones,-2)
            upper_onesL=tf.linalg.band_part(ones_plusdim,num_lower=0,num_upper=-1)
            upper_onesL=x.factory.tensor(upper_onesL)

            """
            1, 1, 1
            0, 1, 1
            0, 0, 1
            """



            diff_matrixL = 2 * tf.linalg.diag(ones) -tf.linalg.band_part(ones_plusdim, num_lower=0, num_upper=1)
            diff_matrixL=x.factory.tensor(diff_matrixL)

            """
             1,-1, 0, 0
             0, 1,-1, 0
             0, 0, 1,-1
             0, 0, 0, 1
            """



        with tf.device(self.server_1.device_name):
            y_bits=y.bits()    # 从低位到高位
            ny_bits=1-y_bits

            ones = tf.ones_like(y_bits.to_native())
            ones_plusdim = tf.expand_dims(ones, -1) * tf.expand_dims(ones, -2)
            upper_onesR = tf.linalg.band_part(ones_plusdim, num_lower=0, num_upper=-1)
            upper_onesR =y.factory.tensor(upper_onesR)

            diff_matrixR = 2 * tf.linalg.diag(ones) - tf.linalg.band_part(ones_plusdim, num_lower=0, num_upper=1)
            diff_matrixR= y.factory.tensor(diff_matrixR)

        left_leq_right_bits=self.left_leq_right_bits(x_bits, y_bits, output_modulus)


        left_neq_right_bits=self.left_equal_right_bits(x_bits, ny_bits,output_modulus)
        left_neq_right_bits_expand=left_neq_right_bits.expand_dims(axis=-1)
        #lower_ones=PondPublicTensor(self, value_on_0=lower_onesL, value_on_1=lower_onesR, is_scaled=False)
        #diff_matrix=PondPublicTensor(self, value_on_0=diff_matrixL, value_on_1=diff_matrixR, is_scaled=False)


        with tf.device(self.server_0.device_name):
            accumulated_neqL = upper_onesL.matmul(left_neq_right_bits_expand.share0)

        with tf.device(self.server_1.device_name):
            accumulated_neqR = upper_onesR.matmul(left_neq_right_bits_expand.share1)  #  2,2,2, 1, 1, 1, 0, 0, 0

        #--------------------------------------------------------------------

        #accumulated_neq=lower_ones.matmul(left_neq_right_bits)  # 0,0,0,1, 1, 1,1, 2,2,2,...
        accumulated_neq=PondPrivateTensor(self, share0=accumulated_neqL, share1=accumulated_neqR, is_scaled=False)

        print("accumulated_neq:",accumulated_neq)
        accumulated_neq_nzero=self.euqal_zero_small(accumulated_neq,output_modulus)  # 0, 0, 0, 0, 0, 0, 1, 1, 1
        accumulated_neq_iszero=1-accumulated_neq_nzero                               # 1, 1, 1, 1, 1, 1, 0, 0, 0

        with tf.device(self.server_0.device_name):
            diff_accumulated_neq_iszeroL=diff_matrixL.matmul(accumulated_neq_iszero.share0)
            diff_accumulated_neq_iszeroL=diff_accumulated_neq_iszeroL.squeeze(axis=-1)
        with tf.device(self.server_1.device_name):
            diff_accumulated_neq_iszeroR=diff_matrixR.matmul(accumulated_neq_iszero.share1)
            diff_accumulated_neq_iszeroR=diff_accumulated_neq_iszeroR.squeeze(axis=-1)

        diff_accumulated_neq_iszero=PondPrivateTensor(self, share0=diff_accumulated_neq_iszeroL, share1=diff_accumulated_neq_iszeroR, is_scaled=False)
            # 0, 0, 0, 0, 0, 1, 0, 0, 0

        # old_shape=diff_accumulated_neq_iszero.shape.as_list()
        # print("old_shape", old_shape, type(old_shape))
        # new_shape=old_shape[0:-1]
        # print("new_shape:", new_shape, type(new_shape))
        # diff_accumulated_neq_iszero = self.reshape(diff_accumulated_neq_iszero, shape=new_shape)
        # #diff_accumulated_neq_iszero=diff_accumulated_neq_iszero.reshape(shape=list(new_shape))

        leq=(left_leq_right_bits*diff_accumulated_neq_iszero).reduce_sum(axis=-1)
        #return left_leq_right_bits
        #return left_neq_right_bits
        #return diff_accumulated_neq_iszero
        return leq





    def left_leq_right2(self,x: AbstractTensor, y: AbstractTensor, output_modulus: int=128) -> PondPrivateTensor:
        """
        这个函数有问题
        :param x: in device 0
        :param y: in device 1
        :return:  1 if x<=y else 0 in PondPrivateTensor
        """
        with tf.device(self.server_0.device_name):
            x_bits=x.bits()
            #x_bits_expand=tf.expand_dims(x_bits.to_native,axis=-1)
            ones=tf.ones_like(x_bits.to_native())


            ones_plusdim=tf.expand_dims(ones,-1)*tf.expand_dims(ones,-2)
            upper_onesL=tf.linalg.band_part(ones_plusdim,num_lower=0,num_upper=-1)
            upper_onesL=x.factory.tensor(upper_onesL)

            """
            1, 1, 1
            0, 1, 1
            0, 0, 1
            """



            diff_matrixL = 2 * tf.linalg.diag(ones) -tf.linalg.band_part(ones_plusdim, num_lower=0, num_upper=1)
            diff_matrixL=x.factory.tensor(diff_matrixL)

            """
             1,-1, 0, 0
             0, 1,-1, 0
             0, 0, 1,-1
             0, 0, 0, 1
            """



        with tf.device(self.server_1.device_name):
            y_bits=y.bits()    # 从低位到高位
            ny_bits=1-y_bits

            ones = tf.ones_like(y_bits.to_native())
            ones_plusdim = tf.expand_dims(ones, -1) * tf.expand_dims(ones, -2)
            upper_onesR = tf.linalg.band_part(ones_plusdim, num_lower=0, num_upper=-1)
            upper_onesR =y.factory.tensor(upper_onesR)

            diff_matrixR = 2 * tf.linalg.diag(ones) - tf.linalg.band_part(ones_plusdim, num_lower=0, num_upper=1)
            diff_matrixR= y.factory.tensor(diff_matrixR)

        left_leq_right_bits=self.left_leq_right_bits(x_bits, y_bits, output_modulus)


        left_neq_right_bits=self.left_equal_right_bits(x_bits, ny_bits,output_modulus)
        left_neq_right_bits_expand=left_neq_right_bits.expand_dims(axis=-1)
        #lower_ones=PondPublicTensor(self, value_on_0=lower_onesL, value_on_1=lower_onesR, is_scaled=False)
        #diff_matrix=PondPublicTensor(self, value_on_0=diff_matrixL, value_on_1=diff_matrixR, is_scaled=False)

        #
        # with tf.device(self.server_0.device_name):
        #     accumulated_neqL = upper_onesL.matmul(left_neq_right_bits_expand.share0)
        #
        # with tf.device(self.server_1.device_name):
        #     accumulated_neqR = upper_onesR.matmul(left_neq_right_bits_expand.share1)  #  2,2,2, 1, 1, 1, 0, 0, 0
        #
        #
        # #--------------------------------------------------------------------
        #

        # accumulated_neq=PondPrivateTensor(self, share0=accumulated_neqL, share1=accumulated_neqR, is_scaled=False)

        accumulated_neq=self.cumsum(x=left_neq_right_bits_expand,axis=-1, reverse=True)


        print("accumulated_neq:",accumulated_neq)
        accumulated_neq_nzero=self.euqal_zero_small(accumulated_neq,output_modulus)  # 0, 0, 0, 0, 0, 0, 1, 1, 1
        accumulated_neq_iszero=1-accumulated_neq_nzero                               # 1, 1, 1, 1, 1, 1, 0, 0, 0

        with tf.device(self.server_0.device_name):
            diff_accumulated_neq_iszeroL=diff_matrixL.matmul(accumulated_neq_iszero.share0)
            diff_accumulated_neq_iszeroL=diff_accumulated_neq_iszeroL.squeeze(axis=-1)
        with tf.device(self.server_1.device_name):
            diff_accumulated_neq_iszeroR=diff_matrixR.matmul(accumulated_neq_iszero.share1)
            diff_accumulated_neq_iszeroR=diff_accumulated_neq_iszeroR.squeeze(axis=-1)

        diff_accumulated_neq_iszero=PondPrivateTensor(self, share0=diff_accumulated_neq_iszeroL, share1=diff_accumulated_neq_iszeroR, is_scaled=False)
            # 0, 0, 0, 0, 0, 1, 0, 0, 0

        # old_shape=diff_accumulated_neq_iszero.shape.as_list()
        # print("old_shape", old_shape, type(old_shape))
        # new_shape=old_shape[0:-1]
        # print("new_shape:", new_shape, type(new_shape))
        # diff_accumulated_neq_iszero = self.reshape(diff_accumulated_neq_iszero, shape=new_shape)
        # #diff_accumulated_neq_iszero=diff_accumulated_neq_iszero.reshape(shape=list(new_shape))

        leq=(left_leq_right_bits*diff_accumulated_neq_iszero).reduce_sum(axis=-1)
        #return left_leq_right_bits
        #return left_neq_right_bits
        #return diff_accumulated_neq_iszero
        return leq






    @memoize
    def mod(self, x, y):
        """
        Performs a true division of `x` by `y` where `y` is public.

        No flooring is performing if `y` is an integer type as it is implicitly
        treated as a float.
        """

        assert isinstance(x, PondPrivateTensor)

        if isinstance(y, float):
            #y_inverse = 1. / y
            raise TypeError("Don't know how to divide by type {}".format(type(y)))
        if isinstance(y, int):
            with tf.device(self.server_0.device_name):
                zL=x.share0%y
            with tf.device(self.server_1.device_name):
                zR=x.share1%y

            z=PondPrivateTensor(self, zL, zR, is_scaled=False)
        elif isinstance(y, PondPublicTensor):
            #y_inverse = self.reciprocal(y)
            raise TypeError("Don't know how to divide by type {}".format(type(y)))
        else:
            raise TypeError("Don't know how to divide by type {}".format(type(y)))

        return z



    def geq_zero(self, x: PondPrivateTensor)-> PondPrivateTensor:
        with tf.device(self.server_0.device_name):
            xL_bits=x.share0.bits()
            xL_lower_bits, xL_top_bit=xL_bits.split(num_split=[ int(xL_bits.shape[-1]-1) ,1], axis=-1)

            # ones=tf.ones_like(xL_lower_bits)
            # from_1_to_n=tf.cumsum(ones, axis=-1)
            # from_0_to_n_1=from_1_to_n-ones
            # power2_from_0_to_n_1=tf.pow(2, from_0_to_n_1)

            xL_lower=bits_to_int(xL_lower_bits.value)

            xL_lower=x.share0.factory.tensor(xL_lower)

            high=x.share0.factory.tensor(np.array([1])).left_shift(xL_lower_bits.shape[-1]) # 2^{N-1}
            high_m_xL_lower=high-xL_lower



        with tf.device(self.server_1.device_name):
            xR_bits=x.share1.bits()
            xR_lower_bits, xR_top_bit=xR_bits.split(num_split=[ int(xR_bits.shape[-1])-1 ,1], axis=-1)

            xR_lower=bits_to_int(xR_lower_bits.value)
            xR_lower=x.share1.factory.tensor(xR_lower)


        top_bit_sum=PondPrivateTensor(self, share0=xL_top_bit.squeeze(axis=-1), share1=xR_top_bit.squeeze(axis=-1), is_scaled=False)


        print("high_m_xL_lower:", high_m_xL_lower)
        print("xR_lower:", xR_lower)
        print("top_bit_sum:", top_bit_sum)

        lower_carry=self.left_leq_right(high_m_xL_lower, xR_lower)
        print("lower_carry:", lower_carry)

        z=(top_bit_sum+lower_carry+1)%(2)
        print("z=", z)
        #return PondPrivateTensor(self, share0=high_m_xL_lower, share1=high_m_xL_lower-high_m_xL_lower, is_scaled=False)

        return z





def bits_to_int(x):
    ones = tf.ones_like(x)
    from_1_to_n = tf.cumsum(ones, axis=-1)
    from_0_to_n_1 = from_1_to_n - ones
    power2_from_0_to_n_1 = tf.pow(2, from_0_to_n_1)
    print("power2_from_0_to_n_1:", power2_from_0_to_n_1)
    return tf.reduce_sum(x*power2_from_0_to_n_1, axis=-1)


# def cycle_lshift2(x_reshape, k_reshape):
#     """
#
#     :param x_reshape:   [m, n]
#     :param k_reshape:   [m]  Z/nZ
#     :return:          [m, n]
#     """
#     x_reshape_split = x_reshape.split(num_split=k_reshape.shape[0])
#     k_reshape_split = k_reshape.split(num_split=k_reshape.shape[0])
#     print("x_reshape_split", x_reshape_split)
#     print("k_reshape_split", k_reshape_split)
#
#     x_reshape_shift_split = []
#     for i in range(len(x_reshape_split)):
#         x_reshape_shift_split = x_reshape_shift_split + [
#             tf.roll(x_reshape_split[i].reshape(axes=[-1]).to_native(), 0 - k_reshape_split[i].to_native(), axis=[-1])]
#
#     x_reshape_shift = tf.stack(x_reshape_shift_split, axis=0)
#     x_reshape_shift = x_reshape.factory.tensor(x_reshape_shift)
#     #print("x_reshape_shift:", x_reshape_shift)
#
#     return x_reshape_shift







def cycle_lshift(x_reshape, k_reshape):

    return cycle_rshift(x_reshape, 0-k_reshape)


def cycle_rshift(x_reshape, k_reshape):
    """

    :param x_reshape:   [m, n]
    :param k_reshape:   [m]  Z/nZ
    :return:          [m, n]
    """
    shifted_x_reshape=cycle_rshift_tensor(x_reshape.to_native(),  k_reshape.to_native())
    return x_reshape.factory.tensor(shifted_x_reshape)


def cycle_rshift_tensor1(x_reshape : tf.Tensor, k_reshape: tf.Tensor):
    """
     这个函数有近似性问题
    :param x_reshape:   [m, n]
    :param k_reshape:   [m]  Z/nZ
    :return:          [m, n]
    """
    k_reshape=tf.floormod(k_reshape, tf.shape(x_reshape)[-1])
    x_reshape_fft =  tf.signal.fft(tf.cast(x_reshape,dtype='complex128'))
    k_reshape_fft = tf.signal.fft(tf.one_hot(k_reshape, depth=tf.shape(x_reshape)[-1], axis=-1,dtype='complex128'))
    print("x_reshape_fft", x_reshape_fft)
    print("k_reshape_fft", k_reshape_fft)

    x_reshape_shift_fft=x_reshape_fft*k_reshape_fft
    x_reshape_shift=tf.signal.ifft(x_reshape_shift_fft)

    x_reshape_shift=tf.cast(x_reshape_shift,dtype=x_reshape.dtype)
    #x_reshape_shift=x_reshape.factory.tensor(x_reshape_shift)
    #print("x_reshape_shift:", x_reshape_shift)
    return x_reshape_shift

def cycle_rshift_tensor(x_reshape : tf.Tensor, k_reshape: tf.Tensor):
    """

    :param x_reshape:   [m, n]
    :param k_reshape:   [m]  Z/nZ
    :return:          [m, n]
    """
    range_matrix=tf.matmul(tf.expand_dims(tf.ones_like(k_reshape),axis=-1), tf.expand_dims(tf.range(tf.shape(x_reshape)[-1], dtype='int32' ),axis=0 ))  #[m,n]
    shifted_range_matrix=(-tf.expand_dims(k_reshape, axis=-1)+range_matrix)%tf.shape(x_reshape)[-1] #[m, n]
    shift_matrix=tf.one_hot(indices=shifted_range_matrix,depth=tf.shape(x_reshape)[-1] ,dtype='int32') #[m,n,n]

    x_reshape_shift=tf.matmul(shift_matrix, tf.expand_dims(x_reshape,axis=-1) )
    x_reshape_shift=tf.squeeze(x_reshape_shift, axis=-1)

    return x_reshape_shift


if __name__=='__main__':
    print("test")

    x=tf.constant(value=[[0,1,2,3],[4,5,6,7]])
    i=tf.constant(value=[1,3])
    xi=cycle_rshift_tensor(x,i)

    sess=tf.Session()
    print(sess.run(xi))



