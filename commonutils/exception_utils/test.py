# encoding: utf-8
"""
@author: guanshun
@contact: yashun.zys@alibaba-inc.com
@time: 2020-01-13 14:29
@file: test.py
@desc: 模拟程序引入resultcode包 实现自定义报错
"""

# import copy
# import numpy as np
from commonutils.exception_utils.result_code import result_code
from commonutils.exception_utils.exception_utils import MorseException

# 调用方直接引用result_code，无需关心language
# language配置在result_code.py里面通过全局配置Config获取的。

try:
    # 模拟程序运行
    a = 1
    b = 2
    # 输出msg
    print "msg:", result_code.VERIFY_TIMESTAMP_ERROR.get_message()
    # 输出code
    print "code:", result_code.VERIFY_TIMESTAMP_ERROR.get_code()
    # 程序运行到这里报错了
    testcase = 'case2'
    # case 1
    if testcase == "case1":
        raise MorseException(result_code.PARAM_NONE_ERROR, param="conf")

    # case 2
    if testcase == 'case2':
        shape = [(1, 3), (2, 3)]
        raise MorseException(result_code.MODEL_MPC_EXBYTE_ERROR, ex=10, shape=shape)

except MorseException as e:
    print "errmsg:", e.get_message()
    print "errmsg:", e.get_msg()
    print "description:", e.get_description()
    print str(e)
except Exception as e:
    print 33
    print str(e)
