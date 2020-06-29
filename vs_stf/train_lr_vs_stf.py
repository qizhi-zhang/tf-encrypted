#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : train_lr_vs_stf
   Author : qizhi.zqz
   Email: qizhi.zqz@alibaba-inc.com
   Create Time : 2020-06-20 22:59
   Description : description what the main function of this file
"""

"""Private training on combined data from several data owners"""
import tf_encrypted as tfe
import json
# from common_private import  ModelOwner, LogisticRegression, XOwner, YOwner
from tfe_keeper.common_private import LogisticRegression
from tfe_keeper.read_data_tf import get_data_xy, get_data_x, get_data_y
from tf_encrypted.keras import backend as KE
import tensorflow as tf
# import sys
import time
import platform
import os
from commonutils.common_config import CommonConfig

# if platform.system() == "Darwin":
#     absolute_path = "/Users/qizhi.zqz/projects/TFE/tf-encrypted/examples/test_on_morse_datas/data"
# else:
absolute_path = "/app/data"


def run(tf_config_file=None):

    matchColNum = 1
    featureNumX = 85 #26  # 1198 #291
    featureNumY = 85 #25
    record_num = 59997 #45211  # 30000 #50038

    epoch_num = 1
    batch_size = 256
    learning_rate = 0.01

    #path_x="Bank_Marketing_L.csv"
    path_x="aps_failure_L.csv"
    #path_y="Bank_Marketing_R.csv"
    path_y = "aps_failure_R.csv"
    progress_file="./progress_file"
    path_x = os.path.join(absolute_path, path_x)
    path_y = os.path.join(absolute_path, path_y)

    train_batch_num = epoch_num * record_num // batch_size + 1
    feature_num = featureNumX + featureNumY


    if tf_config_file:
        config = tfe.RemoteConfig.load(tf_config_file)

    else:
        # default to using local config
        config = tfe.LocalConfig([
            'XOwner',
            'YOwner',
            'RS'])

    CommonConfig.http_logger.info("train_lr/run:  config:" + str(config))
    tfe.set_config(config)
    players = ['XOwner', 'YOwner', 'RS']
    servers=tfe.get_config().get_players(players)
    print("servers=",servers)
    prot = tfe.protocol.SecureNN(*servers)
    tfe.set_protocol(prot)
    # session_target = sys.argv[2] if len(sys.argv) > 2 else None

    if (featureNumY == 0):

        x_train = prot.define_local_computation(player='XOwner', computation_fn=get_data_x,
                                                arguments=(batch_size, path_x, featureNumX,
                                                           matchColNum, epoch_num * 2, 3.0, 1))
        y_train = prot.define_local_computation(player='YOwner', computation_fn=get_data_y,
                                                arguments=(batch_size, path_y, matchColNum,
                                                           epoch_num * 2, 1))
    else:
        x_train1, y_train = prot.define_local_computation(player='YOwner', computation_fn=get_data_xy,
                                                          arguments=(batch_size, path_y, featureNumY,
                                                                     matchColNum, epoch_num * 2, 3.0, 1))
        x_train0 = prot.define_local_computation(player='XOwner', computation_fn=get_data_x,
                                                 arguments=(batch_size, path_x, featureNumX, matchColNum,
                                                            epoch_num * 2, 3.0, 1))
        x_train = prot.concat([x_train0, x_train1], axis=1)

    print("x_train:", x_train)
    print("y_train:", y_train)


    model = LogisticRegression(feature_num, learning_rate=learning_rate)



    #save_op = model.save(modelFilePath, modelFileMachine)
    #save_as_plaintext_op = model.save_as_plaintext(modelFilePlainTextPath, modelFileMachine)
    # load_op = model.load(modelFilePath, modelFileMachine)

    #CommonConfig.http_logger.info("save_op:" + str(save_op))
    # with tfe.Session() as sess:
    try:
        sess = KE.get_session()
        # sess.run(tfe.global_variables_initializer(), tag='init')
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())
    except Exception as e:
        CommonConfig.error_logger.exception(
            'global_variables_initializer error, exception msg:{}'.format(str(e)))

    CommonConfig.http_logger.info("start_time:")
    start_time = time.time()
    CommonConfig.http_logger.info("start_time:" + str(start_time))

    CommonConfig.http_logger.info("train_lr/run: x_train:" + str(x_train))
    CommonConfig.http_logger.info("train_lr/run: y_train:" + str(y_train))
    CommonConfig.http_logger.info("train_lr/run: train_batch_num:" + str(train_batch_num))

    model.fit(sess, x_train, y_train, train_batch_num, progress_file)

    train_time = time.time() - start_time
    print("train_time=", train_time)

    # print("Saving model...")
    # sess.run(save_op)
    # sess.run(save_as_plaintext_op)
    # print("Save OK.")

    # with open(progress_file, "w") as f:
    #     f.write("1.00")
    #     f.flush()



if __name__ == '__main__':

    # with open('./qqq/conf', 'r') as f:
    #     conf = f.read()
    #     print(conf)
    # conf = conf.replace("True", "true").replace("False", "false")
    # # print(input)
    # conf = json.loads(conf)
    # print(conf)

    import time

    start_time=time.time()
    tf_config_file="/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted/vs_stf/config.json"
    run(tf_config_file)

    end_time=time.time()

    print("time=", end_time-start_time)


    # run(taskId="qqq", conf=conf, modelFileMachine="YOwner",
    # modelFilePath="./qqq/model", modelFilePlainTextPath="./qqq/model/plaintext_model",
    # tf_config_file="/app/file/tfe/qqq/config.json")
