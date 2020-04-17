# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2018 All Rights Reserved.
   ------------------------------------------------------
   File Name : http_util
   Author : huazhong.whz
   Email: huazhong.whz@alibaba-inc.com
   Create Time : 2018/8/6 下午5:10
   Description : description what the main function of this file
"""
import os
#import sys
import time
import json
import requests
from unittest import TestCase, main
# from commonutils.common_config import CommonConfig
from commonutils.http_util import HttpUtil
requests.packages.urllib3.disable_warnings()

class TestBinning(TestCase):
    httpUtil = HttpUtil()

    def test_get_grpc_port(self):
        # get grpc_port
        x = self.httpUtil.post(url="http://172.19.1.218:8080/tfe_keeper/grpc_port", json_data={})
        print(x)

    def test_start_server(self):
        # test start_server
        data = {"taskId":  "qqq", "XOwner" : "127.0.0.1:5677", "YOwner" : "127.0.0.1:5678", "RS" : "127.0.0.1:5679", "Player": "XOwner"}

        x = self.httpUtil.post(url="http://127.0.0.1:8080/tfe_keeper/start_server",json_data=json.dumps(data))
        print(x)

    def test_start_server_inner(self):
        #内网 devU, devV, manager
        data = {"taskId":  "qqq", "xOwner" : "172.19.1.216:5678", "yOwner" : "172.19.1.217:5678", "thirdOwner" : "172.19.1.213:5678", "player": "x_owner"}

        x = self.httpUtil.post(url="http://172.19.1.217:8080/tfe_keeper/start_server",json_data=json.dumps(data))
        print(x)

    def test_start_server_outer(self):
        #公网 devU, devV, researchJ
        data = {"taskId":  "qqq", "xOwner" : "47.102.15.80:8217", "yOwner" : "47.102.15.80:8216", "thirdOwner" : "47.102.15.80:8218", "player": "x_owner"}
        x = self.httpUtil.post(url="http://172.19.1.217:8080/tfe_keeper/start_server",json_data=json.dumps(data))
        print(x)

        data = {"taskId":  "qqq", "xOwner" : "47.102.15.80:8217", "yOwner" : "47.102.15.80:8216", "thirdOwner" : "47.102.15.80:8218", "player": "y_owner"}
        x = self.httpUtil.post(url="http://172.19.1.216:8080/tfe_keeper/start_server",json_data=json.dumps(data))
        print(x)

        data = {"taskId":  "qqq", "xOwner" : "47.102.15.80:8217", "yOwner" : "47.102.15.80:8216", "thirdOwner" : "47.102.15.80:8218", "player": "third_owner"}
        x = self.httpUtil.post(url="http://172.19.1.218:8080/tfe_keeper/start_server",json_data=json.dumps(data))
        print(x)

    def test_train(self):
        # test train
        with open('unittest/qqq/conf', 'r') as f:
            conf=f.read()
            print(conf)
        conf=conf.replace("True","true").replace("False","false")
        # print(input)
        conf = json.loads(conf)
        print(conf)

        data = {"taskId": "qqq", "conf": conf, "modelFileMachine": "YOwner", "modelFilePath": "./qqq/model", "test_flag": True }

        x = self.httpUtil.post(url="http://127.0.0.1:5000/tfe_keeper/train",json_data=json.dumps(data))
        print(x)

    def test_train_and_predict_inner(self):
        # test train_and_predict
        with open('unittest/qqq/conf', 'r') as f:
            conf = f.read()
            print(conf)
        conf = conf.replace("True","true").replace("False","false")
        # print(input)
        conf = json.loads(conf)
        print(conf)

        data = {"taskId": "qqq", "conf": conf, "modelFileMachine": "y_owner", "modelFilePath": "file/qqq/model",
                "modelName": "model", "test_flag": False}  # 相对路径
        #
        x = self.httpUtil.post(url="http://172.19.1.216:8080/tfe_keeper/train_and_predict", json_data=json.dumps(data))
        print(x)

    def test_train_and_predict_outer(self):
        # 公网 devU, devV, researchJ
        with open('unittest/qqq/conf', 'r') as f:
            conf = f.read()
            print(conf)
        conf = conf.replace("True","true").replace("False","false")
        # print(input)
        conf = json.loads(conf)
        print(conf)
        data = {"taskId": "qqq", "conf": conf, "modelFileMachine": "YOwner", "modelFilePath": "./qqq/model", "test_flag": True }

        x=self.httpUtil.post(url="http://172.19.1.216:8080/tfe_keeper/train_and_predict",json_data=json.dumps(data))
        print(x)

    def test_check_progress(self):
        # # check_progress
        #
        data = {"taskId": "qqq", "taskType": "train" }
        x = self.httpUtil.post(url="http://127.0.0.1:5000/tfe_keeper/check_progress", json_data=json.dumps(data))
        print(x)

    def test_predict(self):
        # predict
        with open('unittest/qqq/conf', 'r') as f:
            conf = f.read()
            print(conf)
        conf = conf.replace("True","true").replace("False","false")
        # print(input)
        conf = json.loads(conf)
        data = {"taskId": "qqq", "conf": conf, "modelFileMachine": "YOwner", "modelFilePath": "./qqq/model", "test_flag": True,  }

        x = self.httpUtil.post(url="http://127.0.0.1:5000/tfe_keeper/predict",json_data=json.dumps(data))
        print(x)

    def test_check_progress_predict(self):
        # check_progress

        data = {"taskId": "qqq", "taskType": "predict" }
        x = self.httpUtil.post(url="http://127.0.0.1:5000/tfe_keeper/check_progress", json_data=json.dumps(data))
        print(x)

    def test_kill_server(self):
        # test kill server
        data = {"taskId":  "qqq"}
        x = self.httpUtil.post(url="http://127.0.0.1:8080/tfe_keeper/kill_server",json_data=json.dumps(data))
        print(x)


if __name__ == '__main__':
    main()
