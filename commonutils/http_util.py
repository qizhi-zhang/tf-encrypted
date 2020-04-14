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
import sys
import time
import json
import requests
#from commonutils.common_config import CommonConfig

requests.packages.urllib3.disable_warnings()


class HttpUtil(object):
    _session = None
    MB = 1024 ** 2

    @staticmethod
    def get_request_session():
        """
            requests session pool
            key: cu_id:cu_version
            value: sessoin
        """
        if HttpUtil._session is None:
            return requests.Session()

    @staticmethod
    def get(url, dic_data, time_out=10):
        try:
            begin_time = time.time()
            session = HttpUtil.get_request_session()
            # headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
            # # requests.packages.urllib3.disable_warnings()
            response = session.get(url, dic_data, timeout=time_out, verify=False)
            end_time = time.time()
            millis = int((end_time - begin_time) * 1000)
            # CommonConfig.http_logger.info(
            #    '[{}]elapsed {} millis, input_size[{}], status[{}]'.format(url, millis, sys.getsizeof(dic_data),
            #                                                               response.status_code))
            # 返回字典类型
            return response.json()
        except Exception as e:
            end_time = time.time()
            millis = int((end_time - begin_time) * 1000)
            # CommonConfig.error_logger.exception('http post error:[{}], , cost:{}, exception msg:{}'.format(
            #     url, millis, str(e)))
            return None

    @staticmethod
    def post(url, json_data, time_out=10):
        try:
            begin_time = time.time()
            session = HttpUtil.get_request_session()
            headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
            # requests.packages.urllib3.disable_warnings()
            response = session.post(url, data=json_data, headers=headers, timeout=time_out, verify=False)
            end_time = time.time()
            millis = int((end_time - begin_time) * 1000)
            # CommonConfig.http_logger.info(
            #     '[{}]elapsed {} millis, input_size[{}], status[{}]'.format(url, millis, sys.getsizeof(json_data),
            #                                                               response.status_code))
            # 返回字典类型
            return response.json()
        except Exception as e:
            end_time = time.time()
            millis = int((end_time - begin_time) * 1000)
            # CommonConfig.error_logger.exception(
            #     'http post error:[{}], , cost:{}, exception msg:{}'.format(url, millis, str(e)))
            return None
            # raise Exception("http post error,url:{},error msg:{}".format(url, e.message))

    @staticmethod
    def post_byte_data(url, data, param_dic, time_out=10):
        """
        post请求同时发送压缩数据和普通业务参数
        :param url:
        :param data: byte数据
        :param param_dic: 普通业务参数，dict类型
        :return:
        """
        try:
            begin_time = time.time()
            session = HttpUtil.get_request_session()
            headers = {'Content-type': 'application/zip'}
            response = session.post(url=url, data=data, headers=headers, params=param_dic, timeout=time_out)
            end_time = time.time()
            millis = int((end_time - begin_time) * 1000)
            unique_id = json.loads(param_dic).get('uniqueId')
            step = json.loads(param_dic).get('step')
            # CommonConfig.http_logger.info(
            #     '[{}]elapsed {} millis, input_size[{}], status[{}], id:{}, step:{}'.format(
            #        url, millis, sys.getsizeof(data), response.status_code, unique_id, step))

            return response.json()
        except Exception as e:
            # CommonConfig.error_logger.exception('http post error:[{}], input[], param_dic:{} exception msg:{}'.format(
            #     url, param_dic, str(e)))
            # # res = dict()
            # res['status'] = False
            # res['errMsg'] = e
            return None

    @staticmethod
    def post_file(url, file_name, param_dic):
        try:
            begin_time = time.time()
            # dataCipherFilePath = os.path.join(os.getcwd(), file_name)  # file_name = 'aaa/bbb/aaaa.csv'
            files = {'file': (os.path.basename(file_name), open(file_name, 'rb'))}
            response = requests.post(url, data=param_dic, files=files, timeout=10 * 60)  # 10m
            end_time = time.time()
            millis = int((end_time - begin_time) * 1000)
            content = response.content
            # CommonConfig.http_logger.info(
            #     '[{}]elapsed {} millis, input_size[{}], status[{}]'.format(url, millis, sys.getsizeof(files),
               #                                                            response.status_code))
            # print content.json()
            # print json.loads(content)
            return json.loads(content)
        except Exception as e:
            #traceback.print_exc(e)
            # CommonConfig.error_logger.exception('http post error:[{}], input[], param_dic:{} exception msg:{}'.format(
            #     url, param_dic, str(e)))
            # #return None
            print(e)

    @staticmethod
    def post_to_get_file(url, json_data, time_out=10):
        try:
            begin_time = time.time()
            session = HttpUtil.get_request_session()
            headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
            # requests.packages.urllib3.disable_warnings()
            response = session.post(url, data=json_data, headers=headers, timeout=time_out, verify=False)
            end_time = time.time()
            millis = int((end_time - begin_time) * 1000)
            # CommonConfig.http_logger.info(
            #     '[{}]elapsed {} millis, input_size[{}], status[{}]'.format(url, millis, sys.getsizeof(json_data),
            #                                                               response.status_code))
            # 返回字典类型
            return response
        except Exception as e:
            end_time = time.time()
            millis = int((end_time - begin_time) * 1000)
            # CommonConfig.error_logger.exception('http post error:[{}], , cost:{}, exception msg:{}'.format(
            #     url, millis, str(e)))
            #return None
            raise Exception("http post error,url:{},error msg:{}".format(url, e.message))


if __name__=='__main__':
    httpUtil=HttpUtil()


    # get grpc_port
    x = httpUtil.post(url="http://172.19.1.218:8080/tfe_keeper/grpc_port", json_data={})
    print(x)

    # test start_server
    data={"taskId":  "qqq", "XOwner" : "127.0.0.1:5677", "YOwner" : "127.0.0.1:5678", "RS" : "127.0.0.1:5679", "Player": "XOwner"}

    x=httpUtil.post(url="http://127.0.0.1:8080/tfe_keeper/start_server",json_data=json.dumps(data))
    print(x)

    #内网 devU, devV, manager
    #data={"taskId":  "qqq", "xOwner" : "172.19.1.216:5678", "yOwner" : "172.19.1.217:5678", "thirdOwner" : "172.19.1.213:5678", "player": "x_owner"}
    #
    # x=httpUtil.post(url="http://172.19.1.217:8080/tfe_keeper/start_server",json_data=json.dumps(data))
    # print(x)


    #公网 devU, devV, researchJ
    data={"taskId":  "qqq", "xOwner" : "47.102.15.80:8217", "yOwner" : "47.102.15.80:8216", "thirdOwner" : "47.102.15.80:8218", "player": "x_owner"}

    x=httpUtil.post(url="http://172.19.1.217:8080/tfe_keeper/start_server",json_data=json.dumps(data))
    print(x)


    data={"taskId":  "qqq", "xOwner" : "47.102.15.80:8217", "yOwner" : "47.102.15.80:8216", "thirdOwner" : "47.102.15.80:8218", "player": "y_owner"}

    x=httpUtil.post(url="http://172.19.1.216:8080/tfe_keeper/start_server",json_data=json.dumps(data))
    print(x)


    data={"taskId":  "qqq", "xOwner" : "47.102.15.80:8217", "yOwner" : "47.102.15.80:8216", "thirdOwner" : "47.102.15.80:8218", "player": "third_owner"}

    x=httpUtil.post(url="http://172.19.1.218:8080/tfe_keeper/start_server",json_data=json.dumps(data))
    print(x)



    # test train

    # with open('./qqq/conf', 'r') as f:
    #     conf=f.read()
    #     print(conf)
    # conf=conf.replace("True","true").replace("False","false")
    # #print(input)
    # conf=json.loads(conf)
    # print(conf)
    #
    # data={"taskId": "qqq", "conf": conf, "modelFileMachine": "YOwner", "modelFilePath": "./qqq/model", "test_flag": True }
    #
    # x=httpUtil.post(url="http://127.0.0.1:5000/tfe_keeper/train",json_data=json.dumps(data))
    # print(x)







    #test train_and_predict

    # with open('./qqq/conf', 'r') as f:
    #     conf=f.read()
    #     print(conf)
    # conf=conf.replace("True","true").replace("False","false")
    # #print(input)
    # conf=json.loads(conf)
    # print(conf)
    #
    # data = {"taskId": "qqq", "conf": conf, "modelFileMachine": "y_owner", "modelFilePath": "file/qqq/model",
    #         "modelName": "model", "test_flag": False}  # 相对路径
    # #
    # x = httpUtil.post(url="http://172.19.1.216:8080/tfe_keeper/train_and_predict", json_data=json.dumps(data))
    # print(x)

    # 公网 devU, devV, researchJ
    # data={"taskId": "qqq", "conf": conf, "modelFileMachine": "YOwner", "modelFilePath": "./qqq/model", "test_flag": True }
    #
    # x=httpUtil.post(url="http://172.19.1.216:8080/tfe_keeper/train_and_predict",json_data=json.dumps(data))
    # print(x)


    # # check_progress
    #
    # data={"taskId": "qqq", "taskType": "train" }
    # x = httpUtil.post(url="http://127.0.0.1:5000/tfe_keeper/check_progress", json_data=json.dumps(data))
    # print(x)



    # predict

    # data={"taskId": "qqq", "conf": conf, "modelFileMachine": "YOwner", "modelFilePath": "./qqq/model", "test_flag": True,  }
    #
    # x=httpUtil.post(url="http://127.0.0.1:5000/tfe_keeper/predict",json_data=json.dumps(data))
    # print(x)



    # check_progress

    # data={"taskId": "qqq", "taskType": "predict" }
    # x = httpUtil.post(url="http://127.0.0.1:5000/tfe_keeper/check_progress", json_data=json.dumps(data))
    # print(x)


    # test kill server
    # data={"taskId":  "qqq"}
    # x=httpUtil.post(url="http://127.0.0.1:8080/tfe_keeper/kill_server",json_data=json.dumps(data))
    # print(x)


