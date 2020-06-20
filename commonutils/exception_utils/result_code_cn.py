# encoding: utf-8
"""
@author: guanshun
@contact: yashun.zys@alibaba-inc.com
@time: 2020-01-13 14:32
@file: result_code_cn.py
@desc:
"""


from commonutils.exception_utils.exception_utils import ResultCode


VERIFY_TIMESTAMP_ERROR = ResultCode("ME151TFEE0001", "时间戳错误", "时间戳错误")
PARAM_NONE_ERROR = ResultCode("ME151MODE1001", "参数为空", "参数为空")
