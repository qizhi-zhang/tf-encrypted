# pylint: disable=missing-docstring
#import unittest

import numpy as np
import tensorflow as tf
import time
import datetime


import tf_encrypted as tfe
from tf_encrypted.protocol.pond import PondPublicTensor,PondPrivateTensor
from tf_encrypted.tensor import int64factory, int100factory, native_factory
from tf_encrypted.tensor import fixed100, fixed100_ni
from morse_compress import Morse,cycle_rshift, cycle_lshift, bits_to_int
from tf_encrypted.tensor.native import AbstractTensor
import sys



if len(sys.argv) >= 2:
  # config file was specified
  config_file = sys.argv[1]
  config = tfe.RemoteConfig.load(config_file)
else:
  # default to using local config
  config = tfe.LocalConfig([
      'server0',
      'server1',
      'server2',
      'XOwner',
      'YOwner'
  ])
tfe.set_config(config)
players = ['server0', 'server1', 'server2']
morse = Morse(*tfe.get_config().get_players(players))
tfe.set_protocol(morse)
session_target = sys.argv[2] if len(sys.argv) > 2 else None





class TestMorse():

  def test_encode(self):

    with tf.Graph().as_default():


      # expected = np.array([1234567.9875])
      # x = prot.define_constant(expected)
      #
      # print("x.backing_dtype:", x.backing_dtype)
      # print("x.value_on_0.factory",x.value_on_0.factory)

      #morse=Morse()

      ZZ128=native_factory(np.int32, 128)

      x = np.random.randint(128, size=[2, 3])
      #x=np.array([[1,0],[0,1]])
      print("x=", x)
      y = np.random.randint(128, size=[2, 3])
      #y=np.array([[0,1],[1,0]])
      print("y=", y)

      leq=(x<=y)

      x=ZZ128.tensor(x)


      y=ZZ128.tensor(y)







      #
      ZZ3=native_factory(np.int32,3)
      j=np.random.randint(6, size=[2 ])
      print("j=", j)
      j=ZZ3.tensor(j)

      #print("y=",y)

      yj=morse.assistant_OT(y,j)
      #
      #
      # yj2=morse.random_assistant_OT(y,j)

      is_equal_small = morse.left_equal_right_small(x, y, output_modulus=128)
      is_equal=morse.left_equal_right(x, y,output_modulus=128)

      #is_equal2=morse.equal_zero(PondPrivateTensor(prot, share0=y, share1=0-y, is_scaled=False))

      left_leq_right=morse.left_leq_right(x,y)
      with tfe.Session() as sess:
        #actual = sess.run(x)
        #np.testing.assert_array_almost_equal(actual, expected, decimal=3)


        print(time.time())
        print("sess.run(y) start")
        print(sess.run(y))

        print(time.time())
        print("yj start")
        print("yj:", sess.run(yj))




        print(time.time())
        print("is_equal_small start")
        print("is_equal_small:", sess.run(is_equal_small.unwrapped))


        print(time.time())
        print("is_equal start")
        print("is_equal:", sess.run(is_equal.unwrapped))





        print(time.time())
        print("is_equal end, leq start")
        print("left_leq_right:", sess.run(left_leq_right.unwrapped))
        print(time.time())

        print("x<=y:", leq)

class Testbits_to_int():
    def test_bits_to_int(self):
        x=tf.constant([1,0,0,1])
        y=bits_to_int(x)

        with tf.Session() as sess:
            print(sess.run(y))


class Testgeq0():
    def test_geq0(self):
        ZZ128 = native_factory(np.int32, 128 )
        x = np.array(range(-50, 50)).reshape(20,5)
        print("x=", x)



        #morse = Morse()


        zero=ZZ128.tensor(np.zeros_like(x))
        x = ZZ128.tensor(x)

        # i=np.array(range(-10,10))
        # print("i=", i)
        # ZZ5 = native_factory(np.int32, 5)
        # i = ZZ5.tensor(i)
        #
        # shifted_x=cycle_lshift(x, i)

        #shifted_x2=cycle_lshift2(x,i)

        x_leq_0=morse.left_leq_right(x,zero)


        x=PondPrivateTensor(morse, share0=x, share1= zero, is_scaled=False)

        xgeq0_morse=morse.geq_zero(x)

        # xgeq0_Pond=morse.negative(x)

        xgeq0_sercurNN=morse.non_negative(x)

        with tfe.Session() as sess:
            #print("shifted_x=",shifted_x, sess.run(shifted_x))
            #print("shifted_x2=",shifted_x2, sess.run(shifted_x2))


            # print("x>=0 Pond")
            # time0=datetime.datetime.now()
            # xgeq0=sess.run(xgeq0_Pond.unwrapped)
            # print( (np.array(xgeq0[0])+np.array(xgeq0[1]))%2)
            # time1 = datetime.datetime.now()
            # print((time1 - time0).microseconds)

            time_SercurNN=0
            time_morse=0
            for i in range(100):
                print("x>=0 SercurNN")
                time0=datetime.datetime.now()
                xgeq0=sess.run(xgeq0_sercurNN.unwrapped)
                print( (np.array(xgeq0[0])+np.array(xgeq0[1]))%2)
                time1 = datetime.datetime.now()
                print((time1 - time0).total_seconds())
                if i>=1:
                    time_SercurNN=time_SercurNN+(time1 - time0).total_seconds()


                print("x>=0 Morse")
                time0=datetime.datetime.now()
                xgeq0=sess.run(xgeq0_morse.unwrapped)
                print( (np.array(xgeq0[0])+np.array(xgeq0[1]))%2)
                time1 = datetime.datetime.now()
                print((time1 - time0).total_seconds())
                if i>=1:
                    time_morse=time_morse+(time1 - time0).total_seconds()
                #print("x_leq_0:",sess.run(x_leq_0.unwrapped))

            print("avg time_SercurNN=",time_SercurNN/9)
            print("avg time_morse=", time_morse/9)



if __name__ == '__main__':
    #unittest.main()
    test_geq0=Testgeq0()
    test_geq0.test_geq0()

    # test=Testbits_to_int()
    # test.test_bits_to_int()


