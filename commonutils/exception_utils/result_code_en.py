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

# TFE Keeper  1001+  9999-   ME15*TFEE****
DETECT_IDLE_ERROR = ResultCode("ME151TFEE1001", "detect idle error")
START_SERVER_ERROR = ResultCode("ME151TFEE1002", "start server error")
TRAIN_ERROR = ResultCode("ME151TFEE1003", "train error")
PREDICT_ERROR = ResultCode("ME151TFEE1004", "predict error")
TRAIN_AND_PREDICT_ERROR = ResultCode("ME151TFEE1005", "train_and_predict error")
CHECK_PROGRESS_ERROR = ResultCode("ME151TFEE1006", "check progress error")
KILL_SERVER_ERROR = ResultCode("ME151TFEE1007", "kill server error")



PARAM_ERROR = ResultCode("ME151TFEE2001", "param[%(param)s] should not be none")
FILE_NOT_EXIST_ERROR = ResultCode("ME151MODL2002", "file [%(filename)s] not exist")



#MODEL_MPC_EXBYTE_ERROR = ResultCode("ME151MODL1002", "exchange bytes len %(ex)s does not match matrix dim %(shape)s")

