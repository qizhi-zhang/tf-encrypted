# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2018 All Rights Reserved.
   ------------------------------------------------------
   File Name : AppConfig
   Author : huazhong.whz
   Email: huazhong.whz@alibaba-inc.com
   Create Time : 2018/8/6 下午5:45
   Description : description what the main function of this file
"""
import os
import logging
from commonutils.log_util import LogUtil


class CommonConfig(object):
    # 日志文件相关配置
    log_level_config = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARN,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    log_level = log_level_config.get(os.environ.get('log_level')) if os.environ.get('log_level') else logging.INFO
    default_logger = LogUtil.create_logger('default', log_level)
    error_logger = LogUtil.create_logger('error', log_level)
    http_logger = LogUtil.create_logger('http', log_level)
    bining_log = LogUtil.create_logger('bining', log_level)

    dir_home = os.environ.get('dir_home') if os.environ.get('dir_home') else "/app"
    dir_logs = dir_home + '/logs'

    # redis相关配置
    redis_url = os.environ.get('redis_url')
    redis_port = int(os.environ.get('redis_port') or 6379)
    redis_password = os.environ.get('redis_password')

    mpc_base = int(os.environ.get("mpc_base")) if os.environ.get("mpc_base") else 32

    morse_connect_sgx_server_private_key = 'MIICXQIBAAKBgQDJtRy6d31WGrkAx/2DVoXO3ZA2nyjFT67D5sDCmU7TA3jqjj3Fnm1Z2t86R' \
                                           'FxWMzKrTGMcMGhAsWFJzBOU8FU9N8g7hWS43ca1awA19lHzmd+5ZjWf4kKmWNdh5RNxwvRxFx' \
                                           'SLEdIcMisZESRY0xCcIBJoXyeXuDRv3uPWBHtCYQIDAQABAoGBALbIRGxE83MfbB5lHDn0IfU' \
                                           '/mfulZtDGUFx9spelTWSydNQ4aYm28ujGChtG71W9t2C7K+TTiOV21+6D3ArPbq0a4ZIhtnNt' \
                                           'nzsy1BAuUgIWN524P2zRs372kqBkyY7nh48mFqhgzpQBUnLIc7xygupl8dDL0432miTlcHhil' \
                                           'nDxAkEA6mL5ISX+rZ8qKeZBR+slcFVhOTsGvoQd8/HKjslLA1xkJz0O1hYto1Knh+MWDbD3uP' \
                                           'AVDeOmUpEmJSz2DdFZTwJBANxOs6TIBlSyexbTa7sQvvzkYqARY2PIHeUxFV2F6lNrvi7fZ/4' \
                                           'ux+r6oo+6LctcCNqs6MBf9FGnPRsPUjvhXU8CQAq3A5SUXBQr1o2bzRgwk8GS5aLsI97Jw2TH' \
                                           'hO8KHLfGnX19uRPoZ6WzvZzksLlngauerhe4dH4JzKieaZEwJNkCQE4sYxRqocx2FLVRyh1z4' \
                                           'MFt7Q0tfl4OyYTlONaZyT9WsQKC5azNPsVDsGFdyBgsDTxDNMfmhJRgyo8KjbyPyIMCQQCQxb' \
                                           'OwSvafJdHNAO4UbSq+vYAFYH00a3K97onGEYVUQUsvy8Q0Roq5+ahIbUpW3v6PvgA+GyWsLUt' \
                                           '0PGa/AS1A'
    # morse_connect_sgx_server_public_key = 'MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDJtRy6d31WGrkAx/2DVoXO3ZA2nyjFT67D' \
    #                                       '5sDCmU7TA3jqjj3Fnm1Z2t86RFxWMzKrTGMcMGhAsWFJzBOU8FU9N8g7hWS43ca1awA19lHz' \
    #                                       'md+5ZjWf4kKmWNdh5RNxwvRxFxSLEdIcMisZESRY0xCcIBJoXyeXuDRv3uPWBHtCYQIDAQAB'
