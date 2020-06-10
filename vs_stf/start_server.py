#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : start_server
   Author : qizhi.zqz
   Email: qizhi.zqz@alibaba-inc.com
   Create Time : 2020-06-10 21:49
   Description : description what the main function of this file
"""

import tf_encrypted as tfe
import tensorflow as tf
import sys



def _start_server(Player):

    config_file = "/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted/vs_stf/config.json"

    config = tfe.RemoteConfig.load(config_file)
    server = config.server(Player, start=True)
    server.join()


if __name__=='__main__':
    player=sys.argv[1]
    _start_server(Player=player)
