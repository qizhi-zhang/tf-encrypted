# -*- coding: UTF-8 -*-
"""
XOwner (Mayi)：a表, d表   :  a.exposure2 , d.ibnr
YOwner (Baosi)：b表, b1, b2, bc :  b1,  b2,  b.mianpei,   bc

令x=(b2 * (d.ibnr + 1) - least(10000,b.mianpei * (d.ibnr + 1), a.exposure2 * bc))
if b1 AND x>0
        return x
else
        return 0
"""
import datetime
import tf_encrypted as tfe
import tensorflow as tf
import numpy as np
import sys
from tf_encrypted.protocol.pond import PondPublicTensor,PondPrivateTensor
from tf_encrypted.protocol.securenn import SecureNN
from tf_encrypted.tensor import int64factory, int100factory, native_factory
from tf_encrypted.tensor import fixed100, fixed100_ni
from tf_encrypted.tensor.native import AbstractTensor


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
  #
tfe.set_config(config)
players = ['server0', 'server1', 'server2']
seNN = SecureNN(*tfe.get_config().get_players(players))
#tfe.set_protocol(morse)
session_target = sys.argv[2] if len(sys.argv) > 2 else None


bathsize=10000

exposure2 = np.random.normal(size=[bathsize])
ibnr=np.random.normal(size=[bathsize])
b1=1 # or 1
b2=np.random.normal(size=[1])
bc=np.random.normal(size=[1])
mianpei=np.random.normal(size=[bathsize])



# XOwner (Mayi)：a表, d表   :  a.exposure2 , d.ibnr, d.ibnr
def input_exposure2():

    return tf.constant(exposure2,dtype='float32')

def input_ibnr():

    return tf.constant(ibnr,dtype='float32')


# YOwner (Baosi)：b表, b1, b2, bc :   b2,  b.mianpei,   bc

def input_b1():

    return tf.constant(b1, dtype='int32')

def input_b2():

    return tf.constant(b2,dtype='float32')

def input_bc():

    return tf.constant(bc,dtype='float32')

def input_mianpei():

    return tf.constant(mianpei,dtype='float32')

#-------------tf comput -----------------------




def get_least_tf(x,y):
    x=tf.cast(x, 'float32')
    y=tf.cast(y, 'float32')
    return tf.minimum(x,y)


def get_result_tf():
    b1=input_b1()
    b2=input_b2()
    bc=input_bc()
    ibnr=input_ibnr()
    mianpei=input_mianpei()
    exposure2=input_exposure2()


    t1=b2*(ibnr+1)

    t2=mianpei*(ibnr+1)

    t3=exposure2*bc

    least=get_least_tf(10000, t2)
    least=get_least_tf(least, t3)

    x=t1-least

    return tf.cast(b1*tf.cast(x>=0, 'int32'), 'float32')*x
result_tf=get_result_tf()


#--------------------------------------------------------
#--------------tfe compute--------------------------
# XOwner (Mayi)：a表, d表   :  a.exposure2 , d.ibnr, d.ibnr
exposure2=seNN.define_private_input(player="XOwner",inputter_fn=input_exposure2)
ibnr=seNN.define_private_input(player="XOwner",inputter_fn=input_ibnr)

# YOwner (Baosi)：b表, b1, b2, bc :   b2,  b.mianpei,   bc
b1=seNN.define_private_input(player="YOwner", inputter_fn=input_b1)
b2=seNN.define_private_input(player="YOwner", inputter_fn=input_b2)
bc=seNN.define_private_input(player="YOwner", inputter_fn=input_bc)
mianpei=seNN.define_private_input(player="YOwner", inputter_fn=input_mianpei)

#-------------------------------------------------

# 令x=(b2 * (d.ibnr + 1) - least(10000,b.mianpei * (d.ibnr + 1), a.exposure2 * bc))
#          t1                                   t2                      t3
# if b1 AND x>0
#         return x
# else
#         return 0

def get_least(x,y):
    return seNN.greater(x,y)*(y-x)+x


def get_result():
    t1=b2*(ibnr+1)

    t2=mianpei*(ibnr+1)

    t3=exposure2*bc

    least=get_least(10000, t2)
    least=get_least(least, t3)

    x=t1-least

    return seNN.select(b1*seNN.non_negative(x),0,x)
    #return b1*seNN.non_negative(x)*x

result=get_result()

###--------------output -----------------

def get_output(result):
    return tf.print(result)

print_op=seNN.define_output(player="YOwner",arguments=result,outputter_fn=get_output)


sess=tfe.Session()

time0=datetime.datetime.now()

print(sess.run(print_op))

time1=datetime.datetime.now()

print("time of secure compute:", (time1-time0).total_seconds())

print(sess.run(result_tf))

time2=datetime.datetime.now()

print("time of normal compute:", (time2-time1).total_seconds())