"""lipay.com Inc.
   Copyright (c) 2004-2018 All Rights Reserved.
   ------------------------------------------------------
   File Name : test_from_researchJ
   Author : qizhi.zqz
   Email: qizhi.zqz@alibaba-inc.com
   Create Time : 2020.3.19
   Description : test_from_researchJ
"""
import os
import sys
import time
import json
import requests

# from commonutils.common_config import CommonConfig

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
            #         url, millis, sys.getsizeof(data), response.status_code, unique_id, step))

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
            #                                                                response.status_code))
            # print content.json()
            # print json.loads(content)
            return json.loads(content)
        except Exception as e:
            # traceback.print_exc(e)
            # CommonConfig.error_logger.exception('http post error:[{}], input[], param_dic:{} exception msg:{}'.format(
            #     url, param_dic, str(e)))
            # return None
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
            # return None
            # raise Exception("http post error,url:{},error msg:{}".format(url, e.message))


if __name__ == '__main__':
    httpUtil = HttpUtil()

    # test start_server
    data = {"taskId": "qqq", "xOwner": "172.19.1.218:5678", "yOwner": "172.19.1.219:5678",
            "thirdOwner": "172.19.1.214:5678", "player": "third_owner"}

    x = httpUtil.post(url="http://172.19.1.214:8081/tfe_keeper/start_server", json_data=json.dumps(data))
    print(x)

    # test train
    #
    # with open('./qqq/conf', 'r') as f:
    #    conf=f.read()
    #    print(conf)
    # conf=conf.replace("True","true").replace("False","false")
    # print(input)
    # conf=json.loads(conf)
    # print(conf)
    #
    # data={"taskId": "qqq", "conf": conf, "modelFileMachine": "y_owner", "modelFilePath": "file/qqq/model", "test_flag": True } #相对路径
    #
    # x=httpUtil.post(url="http://172.19.1.219:8081/tfe_keeper/train",json_data=json.dumps(data))
    # print(x)

    # # check_progress

    # data={"taskId": "qqq", "taskType": "train" }
    # x = httpUtil.post(url="http://172.19.1.219:8081/tfe_keeper/check_progress", json_data=json.dumps(data))
    # print(x)

    # predict

    # with open('./qqq/conf', 'r') as f:
    #    conf=f.read()
    #    print(conf)
    # conf=conf.replace("True","true").replace("False","false")
    # print(input)
    # conf=json.loads(conf)
    # print(conf)
    # data={"taskId": "qqq", "conf": conf, "modelFileMachine": "y_owner", "modelFilePath": "file/qqq/model", "test_flag": True } #相对路径
    #
    # x=httpUtil.post(url="http://172.19.1.219:8081/tfe_keeper/predict",json_data=json.dumps(data))
    # print(x)

    # check_progress

    # data={"taskId": "qqq", "taskType": "predict" }
    # x = httpUtil.post(url="http://172.19.1.219:8081/tfe_keeper/check_progress", json_data=json.dumps(data))
    # print(x)

    # test kill server
    # data={"taskId":  "qqq"}
    # x=httpUtil.post(url="http://172.19.1.214:8081/tfe_keeper/kill_server",json_data=json.dumps(data))
    # print(x)

