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
from tfe_keeper.tfe_keeper_main import app
requests.packages.urllib3.disable_warnings()

class TestBinning(TestCase):
    httpUtil = HttpUtil()
    def setUp(self):

        app.testing = True
        self.client = app.test_client()
        self.conf = """
        {
    "trainParams": {
        "learningRate": 0.01,
        "maxIter": 2,
        "batchSize": 64,
        "epsilon": 0.0001,
        "regularizationL1": 0,
        "regularizationL2": 1.0,
        "solver": "tfe"
                },
    "midMatchField": {
        "node_id1": ["id1", "date"],
        "node_id2": ["id2", "date_time"]
                },
    "algorithm": "LR",
    "trainSetRate": 0.8,
    "positiveValue": 1,

    "nodes": {
        "node_id1": {
            "merchantAddressModular": "http://127.0.0.1",
            "publicKey": "test_public_key"
        },
        "node_id2": {
            "merchantAddressModular": "http://127.0.0.1",
            "publicKey": "test_public_key"
        }
    },
    "dataSet": {
        "node_id1": {
            "storagePath": "file/qqq/data/embed_op_fea_5w_format_x.csv",
            "fileRecord": 50038,
            "isContainY": False,
            "matchColNum": 2,
            "featureNum": 291
        },
        "node_id2": {
            "storagePath": "file/qqq/data/embed_op_fea_5w_format_y.csv",
            "fileRecord": 50038,
            "isContainY": True,
            "matchColNum": 2,
            "featureNum": 0
        }
    },
    "dataSetPredict": {
        "node_id1": {
            "storagePath": "file/qqq/data/embed_op_fea_5w_format_x.csv",
            "fileRecord": 50038,
            "isContainY": False,
            "matchColNum": 2,
            "featureNum": 291
        },
        "node_id2": {
            "storagePath": "file/qqq/data/embed_op_fea_5w_format_y.csv",
            "fileRecord": 50038,
            "isContainY": True,
            "matchColNum": 2,
            "featureNum": 0
        }
    }
}
       """

    def test_get_grpc_port(self):
        # get grpc_port

        # x = self.httpUtil.post(url="http://172.19.1.218:8080/tfe_keeper/grpc_port", json_data={})
        x = self.client.post("/tfe_keeper/grpc_port", data={}, content_type="application/json")
        print(x)


    def test_start_server(self):
        # test start_server
        data = {"taskId":  "qqq", "XOwner" : "127.0.0.1:5677", "YOwner" : "127.0.0.1:5678", "RS" : "127.0.0.1:5679", "Player": "XOwner"}

        # x = self.httpUtil.post(url="http://127.0.0.1:8080/tfe_keeper/start_server",json_data=json.dumps(data))
        x = self.client.post("/tfe_keeper/start_server", data=json.dumps(data), content_type="application/json")
        print(x)

    def test_train(self):
        # test train
        # with open('unittest/qqq/conf', 'r') as f:
        #     conf=f.read()
        #     print(conf)
        conf=self.conf.replace("True","true").replace("False","false")
        # print(input)
        conf = json.loads(conf)
        print(conf)

        data = {"taskId": "qqq", "conf": conf, "modelFileMachine": "YOwner", "modelFilePath": "./qqq/model", "test_flag": True }

        # x = self.httpUtil.post(url="http://127.0.0.1:5000/tfe_keeper/train",json_data=json.dumps(data))
        x = self.client.post("/tfe_keeper/train",data=json.dumps(data), content_type="application/json")
        print(x)

    # def test_train_and_predict(self):
    #     # test train_and_predict
    #     # with open('unittest/qqq/conf', 'r') as f:
    #     #     conf = f.read()
    #     #     print(conf)
    #     conf = self.conf.replace("True","true").replace("False","false")
    #     # print(input)
    #     conf = json.loads(conf)
    #     print(conf)
    #
    #     data = {"taskId": "qqq", "conf": conf, "modelFileMachine": "y_owner", "modelFilePath": "file/qqq/model",
    #             "modelName": "model", "test_flag": False}  # 相对路径
    #     #
    #     # x = self.httpUtil.post(url="http://172.19.1.216:8080/tfe_keeper/train_and_predict", json_data=json.dumps(data))
    #     x = self.client.post("/tfe_keeper/train_and_predict", data=json.dumps(data), content_type="application/json")
    #     print(x)



    def test_check_progress(self):
        # # check_progress
        #
        time.sleep(2)
        data = {"taskId": "qqq", "taskType": "train" }
        # x = self.httpUtil.post(url="http://127.0.0.1:5000/tfe_keeper/check_progress", json_data=json.dumps(data))
        x = self.client.post("/tfe_keeper/check_progress", data=json.dumps(data), content_type="application/json")
        print(x)

    def test_predict(self):
        # predict
        # with open('unittest/qqq/conf', 'r') as f:
        #     conf = f.read()
        #     print(conf)

        conf = self.conf.replace("True","true").replace("False","false")
        # print(input)
        conf = json.loads(conf)
        data = {"taskId": "qqq", "conf": conf, "modelFileMachine": "YOwner", "modelFilePath": "./qqq/model", "test_flag": True}

        # x = self.httpUtil.post(url="http://127.0.0.1:5000/tfe_keeper/predict",json_data=json.dumps(data))
        x = self.client.post("/tfe_keeper/predict", data=json.dumps(data), content_type="application/json")
        print(x)

    def test_check_progress_predict(self):
        # check_progress
        time.sleep(2)
        data = {"taskId": "qqq", "taskType": "predict" }
        # x = self.httpUtil.post(url="http://127.0.0.1:5000/tfe_keeper/check_progress", json_data=json.dumps(data))
        x = self.client.post("/tfe_keeper/check_progress", data=json.dumps(data), content_type="application/json")
        print(x)

    def test_kill_server(self):
        # test kill server
        data = {"taskId":  "qqq"}
        # x = self.httpUtil.post(url="http://127.0.0.1:8080/tfe_keeper/kill_server",json_data=json.dumps(data))
        x = self.client.post("/tfe_keeper/kill_server",data=json.dumps(data), content_type="application/json")
        print(x)


if __name__ == '__main__':
    main()
