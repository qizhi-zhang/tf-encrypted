# encoding: utf-8
"""
@author: guanshun
@contact: yashun.zys@alibaba-inc.com
@time: 2020-01-13 14:17
@file: result_code_en.py
@desc:
"""


from commonutils.exception_utils.exception_utils import ResultCode


# 通用错误
VERIFY_TIMESTAMP_ERROR = ResultCode("ME151MODL0001", "verify timestamp error", "verify timestamp error")
VERIFY_SGIN_ERROR = ResultCode("ME151MODL0002", "verify sign error", "verify sign error")

PARAM_NONE_ERROR = ResultCode("ME151MODL1001", "param[%(param)s] None error")

# 自定义错误
TEE_SYSTEM_ERROR = ResultCode("ME151MODL9999", "tee-algo system error")


MODEL_SOLVER_ERROR = ResultCode("ME152MODL1001", "param[solver] should be auto_smart/sgd_gc/newton_gc")
MODEL_MPC_EXBYTE_ERROR = ResultCode("ME151MODL1002", "exchange bytes len %(ex)s does not match matrix dim %(shape)s")
FILE_NOT_EXIST_ERROR = ResultCode("ME151MODL1003", "file [%(filename)s] not exist")