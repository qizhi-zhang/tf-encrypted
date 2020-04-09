# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2018 All Rights Reserved.
   ------------------------------------------------------
   File Name : log_util
   Author : huazhong.whz
   Email: huazhong.whz@alibaba-inc.com
   Create Time : 2018/8/6 下午5:47
   Description : description what the main function of this file
"""
import logging
import os
import platform
import time
from logging import Formatter
from logging.handlers import TimedRotatingFileHandler

operating_system_platform = platform.system()
default_logs = "/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted" if operating_system_platform == "Darwin" else "/app"

_LOG_FORMAT = '%(asctime)s %(levelname)s ' \
              '%(module)s.%(funcName)s:%(lineno)d %(message)s'
# _DIR_LOGS = os.environ.get('dir_home') if os.environ.get('dir_home') else "/app"
_DIR_LOGS = os.environ.get('dir_home') if os.environ.get('dir_home') else default_logs
_LOG_FILE_BASE = _DIR_LOGS + '/logs'
_LOG_PROFILE_BASE = _DIR_LOGS + '/profile'  # 建模的性能log
_LOG_LOSS_PROFILE_BASE = _DIR_LOGS + '/lossprofile'  # 损失函数的性能log
_LOG_FILE_SUFFIX = '%Y%m%d'


class LogUtil(object):

    def __init__(self):
        if not os.path.exists(_LOG_FILE_BASE):
            os.makedirs(_LOG_FILE_BASE)

    @staticmethod
    def create_profile_logger(name):
        log = logging.getLogger(name)
        if len(log.handlers) > 0:
            return log

        logging.basicConfig(format=_LOG_FORMAT)
        formatter = Formatter(_LOG_FORMAT)

        handler = SafeRotatingFileHandler(
            '{dir_logs}/{log_name}.log'.format(dir_logs=_LOG_PROFILE_BASE, log_name=name), 'midnight', 1, 15
        )

        handler.setFormatter(formatter)
        handler.suffix = _LOG_FILE_SUFFIX
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        return log

    @staticmethod
    def create_loss_profile_logger(name):
        log = logging.getLogger(name)
        if len(log.handlers) > 0:
            return log

        logging.basicConfig(format=_LOG_FORMAT)
        formatter = Formatter(_LOG_FORMAT)
        handler = SafeRotatingFileHandler(
            '{dir_logs}/{log_name}.log'.format(dir_logs=_LOG_LOSS_PROFILE_BASE, log_name=name), 'midnight', 1, 15
        )

        handler.setFormatter(formatter)
        handler.suffix = _LOG_FILE_SUFFIX
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        return log

    @staticmethod
    def create_logger(name, level=logging.INFO):
        """
        生成log日志
        :param name: 日志名称，目前是在特定的目录下面用morse-{name}.log命名
        :param level: 日志级别:msg above the level will be displayed,
                       DEBUG < INFO < WARN < ERROR < CRITICAL
        """
        log = logging.getLogger(name)
        if len(log.handlers) > 0:
            return log

        logging.basicConfig(format=_LOG_FORMAT)
        formatter = logging.Formatter(_LOG_FORMAT)
        # 日志每天凌晨0点轮转，转轮的日志最多保留15天
        handler = SafeRotatingFileHandler(
            '{dir_logs}/morse-{log_name}.log'.format(dir_logs=_LOG_FILE_BASE, log_name=name), 'midnight', 1, 15
        )

        handler.setFormatter(formatter)
        handler.suffix = _LOG_FILE_SUFFIX
        log.addHandler(handler)
        log.setLevel(level)
        return log


class SafeRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False):
        TimedRotatingFileHandler.__init__(self, filename, when, interval, backupCount, encoding, delay, utc)

    """
    Override doRollover
    lines commanded by "##" is changed
    """

    def doRollover(self):
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens.  However, you want the file to be named for the
        start of the interval, not the current time.  If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.

        Override,   1. if dfn not exist then do rename
                    2. _open with "a" model
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)
        # if os.path.exists(dfn):
        #     os.remove(dfn)

        # Issue 18940: A file may not have been created if delay is True.
        # if os.path.exists(self.baseFilename):
        if not os.path.exists(dfn) and os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, dfn)
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        # if not self.delay:
        self.mode = "a"
        self.stream = self._open()
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        # If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:  # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt
