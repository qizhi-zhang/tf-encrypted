# encoding: utf-8
"""
@author: guanshun
@contact: yashun.zys@alibaba-inc.com
@time: 2019-12-02 19:01
@file: exception_utils.py
@desc: 本util定义了ResultCode的class,以及MorseException. 具体resultCode种类，
       如果是英文在result_code_en.py里面枚举定义
       如果是中文在result_code_cn.py里面枚举定义
       具体例子，见test.py

"""


class ResultCode(object):

    # https://yuque.antfin-inc.com/sslab/morse4/nte7vg 错误码文档
    # 第 1-2 位  : ME
    # 第  3  位  : 版本号目前都是(1)
    # 第  4  位  : 错误级别(1/info, 3/warn, 5/error, 7/fatal)
    # 第  5  位  : 错误类型(1/系统错误[用户解决不了的], 2/业务错误[用户能自行解决的])
    # 第 6-9 位  : 应用系统4个字母的缩写大写，各owner自己定义，比如PROD(prod), MODE(smodel)等
    # 第10-13位  : 自行定义， 0*** 通用的，0001是时间戳错误/0002是验签错误, x*** (x=1~9)各owner自行定义

    # 可以添加get set方法

    def __init__(self, code, message, description=''):
        self.code = code  # 错误码
        self.message = message  # 给用户看的错误信息
        self.description = message  # 给开发/运维看的错误信息
        if description != '':
            self.description = description

    def default_success(self, app):
        self.code = "ME110" + app + "0000"
        self.message = "成功"

    def get_code(self):
        return self.code

    def set_code(self, code):
        self.code = code

    def get_message(self):
        return self.message

    def set_message(self, message):
        self.message = message

    def get_description(self):
        return self.description

    def set_description(self, description):
        self.description = description


class MorseException(Exception):
    """
    user-defined exception class. 可以添加get set方法
    """

    def __init__(self, result_code, **kwargs):

        self.result_code = result_code  # resultCode
        self.code = result_code.get_code()  # 错误码
        self.message = result_code.get_message()  # 错误msg，用户看的
        self.description = result_code.get_description()  # 错误description，开发看的
        self.msg = "[" + result_code.get_code() + "] " + result_code.get_message()

        if kwargs:  # 可以自己添加上下文信息
            # 不加code的msg
            self.message = self.message % kwargs  # message里的key一定要和kwargs的key对应上
            # 加了code的msg
            self.msg = self.msg % kwargs
            # description
            self.description = self.description % kwargs

    def get_result_code(self):
        return self.result_code

    def set_result_code(self, result_code):
        self.result_code = result_code

    def get_code(self):
        return self.code

    def get_message(self):
        return self.message

    def get_description(self):
        return self.description

    def get_msg(self):
        return self.msg

    def set_msg(self, msg):
        self.msg = msg

    def __repr__(self):
        return self.msg

    def __str__(self):
        return self.__repr__()

    def __unicode__(self):
        return self.__repr__()
