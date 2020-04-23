# coding=utf-8
from flask import Flask, request, Blueprint
# from flask import Flask, redirect, url_for, request, Blueprint
# import os
import json
import tensorflow as tf
from tf_encrypted.config import RemoteConfig
from multiprocessing import Process
# import threading
from commonutils.common_config import CommonConfig
# if __name__=='__main__':
#     import train_lr
#     import predict_lr
#     import train_and_predict_lr
# else:
from tfe_keeper import train_lr, predict_lr, train_and_predict_lr
import os
import platform
from commonutils.exception_utils.exception_utils import MorseException
from commonutils.exception_utils.result_code import result_code

absolute_path = None
if platform.system() == "Darwin":
    # os.putenv('absolute_path', "/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted")
    absolute_path = "/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted"
else:
    # os.putenv('absolute_path', "/app")
    absolute_path = "/app/file"

app = Flask(__name__)
# @app.route('/service/<name>')
# def success(name):
#     return 'welcome %s' % name
#
#
# @app.route('/login', methods=['POST', 'GET'])
# def login():
#     if request.method == 'POST':
#         user = request.form['nm']
#         return redirect(url_for('success', name=user))
#     else:
#         user = request.args.get('nm')
#         return redirect(url_for('success', name=user))


tfe_keeper = Blueprint('tfe_keeper', __name__)


@tfe_keeper.route('/grpc_port', methods=['GET', 'POST'])
def get_grpc_port():
    try:
        grpc_port = os.environ.get("grpc_port") or '80'
        status = True
        errorCode = 0
        errorMsg = ""
    except Exception as e:
        grpc_port = 0
        status = False
        errorCode = result_code.GET_GRPC_PORT_ERROR.get_code()
        errorMsg = result_code.GET_GRPC_PORT_ERROR.get_message()
        CommonConfig.error_logger.exception(
            'get_grpc_port error, exception msg:{}'.format(str(request.json), str(e)))
    return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg, "grpcPort": grpc_port})


@tfe_keeper.route('/detect_idle', methods=['GET', 'POST'])
def detect_idle():
    """

    :param ip_host:  xxx.xxx.xxx.xxx:host
    :return:   "idle" or "busy"
    """
    print("detect_idle start")
    try:
        # print("request:", request)
        CommonConfig.default_logger.info("detect_idle request:" + str(request))
        request_params = request.json
        CommonConfig.default_logger.info("detect_idle request_params:" + str(request_params))
        # print("request_params:", request_params)
        ip_host = request_params.get('ipHost')
        CommonConfig.default_logger.info("ip_host:" + str(ip_host))
        status = _detect_idle(ip_host)
        CommonConfig.default_logger.info("status:" + str(status))
        return json.dumps({"status": status})
    except Exception as e:
        status = False
        CommonConfig.error_logger.exception(
            'detelt_idle error on input: {}, exception msg:{}'.format(str(request.json), str(e)))
        errorCode = result_code.DETECT_IDLE_ERROR.get_code()
        errorMsg = result_code.DETECT_IDLE_ERROR.get_message()
    return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})


def _detect_idle(ip_host):
    try:
        cluster = tf.train.ClusterSpec({"detect": [ip_host]})
        CommonConfig.default_logger.info("cluster:" + str(cluster))
        server = tf.train.Server(cluster)
        CommonConfig.default_logger.info("server:" + str(server))
        status = "idle"
        # server.join()
    except Exception as e:
        # print(e)
        CommonConfig.error_logger.exception(
            '_detelt_idle error on input: {}, exception msg:{}'.format(str(ip_host), str(e)))
        status = "busy"
    return status


@tfe_keeper.route('/start_server', methods=['GET', 'POST'])
def start_server():
    """

    :param ip_host:  xxx.xxx.xxx.xxx:host
    :return:
     status:  True or False
     errorCode:
     errorMsg:

    """
    # print("start_server")
    try:
        CommonConfig.default_logger.info("start_server request:" + str(request))
        request_params = request.json
        CommonConfig.default_logger.info("start_server request_params:" + str(request_params))
        task_id = request_params.get('taskId')
        CommonConfig.default_logger.info("task_id:" + str(task_id))

        if task_id is None:
            raise MorseException(result_code.PARAM_ERROR, param="taskId")
        # if other is None:
        #   raise MorseException(result_code.PARAM_ERROR, param="other")

        RS_iphost = request_params.get('RS')
        if RS_iphost is None:
            RS_iphost = request_params.get("thirdOwner")
        CommonConfig.default_logger.info("RS_iphost:" + str(RS_iphost))
        if RS_iphost is None:
            raise MorseException(result_code.PARAM_ERROR, param="RS or thirdOwner")
            # status = False
            # errorCode = result_code.START_SERVER_ERROR.get_code()
            # errorMsg = "nether RS nor thirdOwner are given"
            # return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})

        XOwner_iphost = request_params.get('XOwner')
        if XOwner_iphost is None:
            XOwner_iphost = request_params.get('xOwner')
        CommonConfig.default_logger.info("XOwner_iphost=" + str(XOwner_iphost))
        if XOwner_iphost is None:
            raise MorseException(result_code.PARAM_ERROR, param="xOwner or XOwner")
            # status = False
            # errorCode = result_code.START_SERVER_ERROR.get_code()
            # errorMsg = "nether xOwner nor XOwner are given"
            # return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})

        YOwner_iphost = request_params.get('YOwner')
        if YOwner_iphost is None:
            YOwner_iphost = request_params.get('yOwner')
        if YOwner_iphost is None:
            raise MorseException(result_code.PARAM_ERROR, param="yOwner or YOwner")
            # status = False
            # errorCode = result_code.START_SERVER_ERROR.get_code()
            # errorMsg = "nether yOwner nor YOwner are given"
            # return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})

        Player = request_params.get('Player')
        if Player is None:
            Player = request_params.get('player')
        if Player is None:
            raise MorseException(result_code.PARAM_ERROR, param="Player or player")

        # if YOwner_iphost is None:

            # status = False
            # errorCode = result_code.START_SERVER_ERROR.get_code()
            # errorMsg = "nether Player nor player are given"
            # return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})

        if Player == "x_owner" or Player == "xOwner":
            Player = "XOwner"
        if Player == "y_owner" or Player == "yOwner":
            Player = "YOwner"
        if Player == "third_owner" or Player == 'thirdOwner':
            Player = "RS"

        os.makedirs(os.path.join(absolute_path, "tfe/{task_id}".format(task_id=task_id)), exist_ok=True)
        p = Process(target=_start_server, args=(task_id, XOwner_iphost, YOwner_iphost, RS_iphost, Player))
        # status=_start_server(task_id, XOwner_iphost, YOwner_iphost, RS_iphost, Player)
        p.start()
        p.join(timeout=1)
        CommonConfig.default_logger.info("p.pid" + str(p.pid))
        # print(p.is_alive())
        CommonConfig.default_logger.info("p.exitcode" + str(p.exitcode))
        if p.is_alive():
            # with open(os.path.join(absolute_path, 'tfe/{task_id}/server_pid'.format(task_id=task_id)), 'w') as f:
            with open(os.path.join(absolute_path, 'tfe/server_pid'.format(task_id=task_id)), 'w') as f:
                f.write(str(p.pid))

            status = True
            errorCode = 0
            errorMsg = ""
        else:
            raise MorseException(result_code.START_SERVER_ERROR, server=Player)
            # status = False
            # errorCode = result_code.START_SERVER_ERROR.get_code()
            # errorMsg = "start server faild"
        # print("p.exitcode:", p.exitcode)
        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})
    except MorseException as e:
        status = False
        errorMsg = e.get_message()
        errorCode = e.get_code()
        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})
    except Exception as e:
        CommonConfig.error_logger.exception(
            'start_server error, exception msg:{}'.format(str(e)))
        status = False
        errorCode = result_code.START_SERVER_ERROR.get_code()
        errorMsg = str(e)
        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})


def _start_server(task_id, XOwner_iphost, YOwner_iphost, RS_iphost, Player):
    try:
        config = """
{l}
    "XOwner": "{XOwner_iphost}", 
    "YOwner": "{YOwner_iphost}", 
    "RS": "{RS_iphost}"
{r}
        """.format(l="{", r="}", XOwner_iphost=XOwner_iphost, YOwner_iphost=YOwner_iphost, RS_iphost=RS_iphost)
        CommonConfig.default_logger.info("config:" + str(config))
        os.system("pwd")
        CommonConfig.default_logger.info("absolute_path:" + str(absolute_path))
        CommonConfig.default_logger.info("1. tfe config.json: " +
                                         os.path.join(absolute_path,
                                                      'tfe/{task_id}/config.json'.format(task_id=task_id)))

        with open(os.path.join(absolute_path, 'tfe/{task_id}/config.json'.format(task_id=task_id)), 'w') as f:
            f.write(config)

        CommonConfig.default_logger.info("2. tfe config.json:" +
                                         os.path.join(absolute_path,
                                                      'tfe/{task_id}/config.json'.format(task_id=task_id)))

        config = RemoteConfig.load(os.path.join(absolute_path, 'tfe/{task_id}/config.json'.format(task_id=task_id)))
        server = config.server(Player, start=True)
        server.join()
    except Exception as e:
        CommonConfig.error_logger.exception(
            '_start_server error , exception msg:{}'.format(str(e)))
        # print(e)


@tfe_keeper.route('/train', methods=['GET', 'POST'])
def train():
    """
    input:
        taskId, algorithm, conf, modelFileMachine, modelFilePath
    :return:
        status, 
        errorCode, 
        errorMsg
    """
    CommonConfig.default_logger.info("train")
    try:
        CommonConfig.default_logger.info("train request:" + str(request))
        request_params = request.json
        CommonConfig.default_logger.info("train request_params:" + str(request_params))
        task_id = request_params.get('taskId')
        CommonConfig.default_logger.info("task_id:" + str(task_id))
        # algorithm = request_params.get('algorithm')
        modelFileMachine = request_params.get('modelFileMachine')
        CommonConfig.default_logger.info("modelFileMachine:" + str(modelFileMachine))
        if modelFileMachine == "x_owner" or modelFileMachine == "xOwner" or modelFileMachine == "XOwner":
            modelFileMachine = "XOwner"
        elif modelFileMachine == "y_owner" or modelFileMachine == "yOwner" or modelFileMachine == "YOwner":
            modelFileMachine = "YOwner"
        elif modelFileMachine == "third_owner" or modelFileMachine == "thirdOwner" or modelFileMachine == "RS":
            modelFileMachine = "RS"
        else:
            raise MorseException(result_code.PARAM_ERROR, param='modelFileMachine')

        modelFilePath = request_params.get('modelFilePath')
        modelFilePath = os.path.join(absolute_path, modelFilePath)
        modelName = request_params.get('modelName')
        modelFilePlainTextPath = os.path.join(modelFilePath, modelName)
        conf = request_params.get('conf')

        test_flag = request_params.get('test_flag', False)
        unittest_flag = request_params.get('unittest_flag', False)
        if test_flag:
            tf_config_file = None
        else:
            tf_config_file = os.path.join(absolute_path, "tfe/{task_id}/config.json".format(task_id=task_id))

        # train_lr.run(task_id, conf, modelFileMachine, modelFilePath, tf_config_file=tf_config_file)

        p = Process(target=train_lr.run, args=(task_id, conf, modelFileMachine,
                                               modelFilePath, modelFilePlainTextPath, tf_config_file))
        # p = threading.Thread(target=train_lr.run, args=(task_id, conf, 
        # modelFileMachine, modelFilePath, modelFilePlainTextPath, tf_config_file))
        if not unittest_flag:
            p.start()

        CommonConfig.default_logger.info("train Process pid:" + str(p.pid))

        with open(os.path.join(absolute_path, 'tfe/{task_id}/train_pid'.format(task_id=task_id)), 'w') as f:
            f.write(str(p.pid))

        # CommonConfig.default_logger.info("train Process pid:" + str(p.name))
        # 
        # with open(os.path.join(absolute_path, 'tfe/{task_id}/train_pid'.format(task_id=task_id)), 'w') as f:
        #   f.write(str(p.name))

        status = True
        errorCode = 0
        errorMsg = ""
        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})
    except MorseException as e:
        status = False
        errorMsg = e.get_message()
        errorCode = e.get_code()
        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})
    except Exception as e:
        # print(e)
        CommonConfig.error_logger.exception(
            'train error , exception msg:{}'.format(str(e)))
        status = False
        errorCode = result_code.TRAIN_ERROR.get_code()
        errorMsg = str(e)
        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})


@tfe_keeper.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    input:
        taskId, algorithm, conf, modelFileMachine, modelFilePath
    :return:
        status, 
        errorCode, 
        errorMsg
    """
    # print("predict")
    try:
        CommonConfig.default_logger.info("pridict request:" + str(request))
        request_params = request.json
        CommonConfig.default_logger.info("predict request_params:" + str(request_params))
        task_id = request_params.get('taskId')
        CommonConfig.default_logger.info("task_id:" + str(task_id))
        # algorithm = request_params.get('algorithm')
        modelFileMachine = request_params.get('modelFileMachine')

        if modelFileMachine == "x_owner" or modelFileMachine == "xOwner" or modelFileMachine == "XOwner":
            modelFileMachine = "XOwner"
        elif modelFileMachine == "y_owner" or modelFileMachine == "yOwner" or modelFileMachine == "YOwner":
            modelFileMachine = "YOwner"
        elif modelFileMachine == "third_owner" or modelFileMachine == "thirdOwner" or modelFileMachine == "RS":
            modelFileMachine = "RS"
        else:
            raise MorseException(result_code.PARAM_ERROR, param='modelFileMachine')

        modelFilePath = request_params.get('modelFilePath')
        modelFilePath = os.path.join(absolute_path, modelFilePath)
        conf = request_params.get('conf')
        test_flag = request_params.get('test_flag', False)

        progress_file = os.path.join(absolute_path, "tfe/" + task_id + "/predict_progress")

        if test_flag:
            tf_config_file = None
        else:
            tf_config_file = os.path.join(absolute_path, "tfe/{task_id}/config.json".format(task_id=task_id))

        p = Process(target=predict_lr.run, args=(task_id, conf, modelFileMachine,
                                                 modelFilePath, progress_file, tf_config_file))
        unittest_flag = request_params.get('unittest_flag', False)
        if not unittest_flag:
            p.start()
        CommonConfig.default_logger.info("predict Process pid:" + str(p.pid))
        with open(os.path.join(absolute_path, 'tfe/{task_id}/predict_pid'.format(task_id=task_id)), 'w') as f:
            f.write(str(p.pid))

        status = True
        errorCode = 0
        errorMsg = ""
        predict_file = os.path.join(absolute_path, "tfe/{task_id}/predict".format(task_id=task_id))

    except MorseException as e:
        status = False
        errorMsg = e.get_message()
        errorCode = e.get_code()
        predict_file = os.path.join(absolute_path, "tfe/{task_id}/predict".format(task_id=task_id))

    except Exception as e:
        # print(e)
        CommonConfig.error_logger.exception(
            'predict error , exception msg:{}'.format(str(e)))
        status = False
        errorCode = result_code.PREDICT_ERROR.get_code()
        errorMsg = str(e)
        predict_file = ""
    return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg, "predictFile": predict_file})


@tfe_keeper.route('/train_and_predict', methods=['GET', 'POST'])
def train_and_predict():
    """
    input:
        taskId, algorithm, conf, modelFileMachine, modelFilePath
    :return:
        status, 
        errorCode, 
        errorMsg
    """
    CommonConfig.default_logger.info("predict")
    try:
        CommonConfig.default_logger.info("pridict request:" + str(request))
        request_params = request.json
        CommonConfig.default_logger.info("predict request_params:" + str(request_params))
        task_id = request_params.get('taskId')
        CommonConfig.default_logger.info("task_id:" + str(task_id))
        # algorithm = request_params.get('algorithm')
        modelFileMachine = request_params.get('modelFileMachine')
        CommonConfig.default_logger.info("modelFileMachine:" + str(modelFileMachine))

        if modelFileMachine == "x_owner" or modelFileMachine == "xOwner" or modelFileMachine == "XOwner":
            modelFileMachine = "XOwner"
        elif modelFileMachine == "y_owner" or modelFileMachine == "yOwner" or modelFileMachine == "YOwner":
            modelFileMachine = "YOwner"
        elif modelFileMachine == "third_owner" or modelFileMachine == "thirdOwner" or modelFileMachine == "RS":
            modelFileMachine = "RS"
        else:
            raise MorseException(result_code.PARAM_ERROR, param='modelFileMachine')

        modelFilePath = request_params.get('modelFilePath')
        modelFilePath = os.path.join(absolute_path, modelFilePath)
        modelName = request_params.get('modelName')
        modelFilePlainTextPath = os.path.join(modelFilePath, modelName)
        conf = request_params.get('conf')
        test_flag = request_params.get('test_flag', False)

        # progress_file_predict = os.path.join(absolute_path, "tfe/" + task_id + "/predict_progress")

        if test_flag:
            tf_config_file = None
        else:
            tf_config_file = os.path.join(absolute_path, "tfe/{task_id}/config.json".format(task_id=task_id))
        # predict_lr.run(task_id, conf, modelFileMachine, modelFilePath, progress_file, tf_config_file)

        p = Process(target=train_and_predict_lr.run,
                    args=(task_id, conf, modelFileMachine, modelFilePath,
                          modelFilePlainTextPath, tf_config_file))
        unittest_flag = request_params.get('unittest_flag', False)
        if not unittest_flag:
            p.start()

        with open(os.path.join(absolute_path, 
                               'tfe/{task_id}/train_and_predict_pid'.format(task_id=task_id)), 'w') as f:
            f.write(str(p.pid))

        status = True
        errorCode = 0
        errorMsg = ""
        predict_file = os.path.join(absolute_path,
                                    "tfe/{task_id}/predict".format(task_id=task_id))
        return json.dumps({"status": status, "errorCode": errorCode,
                           "errorMsg": errorMsg, "predictFile": predict_file})
    except Exception as e:
        # print(e)
        CommonConfig.error_logger.exception(
            'predict error, exception msg:{}'.format(str(e)))
        status = False
        errorCode = result_code.TRAIN_AND_PREDICT_ERROR.get_code()
        errorMsg = str(e)
        predict_file = ""
        return json.dumps({"status": status, "errorCode": errorCode,
                           "errorMsg": errorMsg, "predictFile": predict_file})


@tfe_keeper.route('/check_progress', methods=['GET', 'POST'])
def check_progress():
    """
    input: taskId, taskType
    :return:
            status
            executeStatus
            percent
            errorMsg
            errorCode
    """

    def check_pid(pid):
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

    CommonConfig.default_logger.info("check_progress")
    percent = 0.0
    try:
        CommonConfig.default_logger.info("check_progress request:" + str(request))
        request_params = request.json
        CommonConfig.default_logger.info("check_progress request_params:" + str(request_params))
        task_id = request_params.get('taskId')
        CommonConfig.default_logger.info("task_id:" + str(task_id))
        taskType = request_params.get('taskType')
        CommonConfig.default_logger.info("taskType:" + str(taskType))

        if taskType == "train":

            with open(os.path.join(absolute_path, 'tfe/{task_id}/train_pid'.format(task_id=task_id)), 'r') as f:
                pid = f.readline()
            if pid is None or pid == "" or pid == "None":
                raise MorseException(result_code.FILE_IS_EMPTY_ERROR,
                                     filename='tfe/{task_id}/train_pid'.format(task_id=task_id))
            pid = int(pid)
            CommonConfig.default_logger.info("pid=" + str(pid))

            pid_exists = check_pid(pid)

            with open(os.path.join(absolute_path,
                                   'tfe/{task_id}/train_progress'.format(task_id=task_id)), "r") as f:
                lines = f.readlines()
            if lines is None or lines == "" or pid == "None":
                raise MorseException(result_code.FILE_IS_EMPTY_ERROR,
                                     filename="tfe/{task_id}/train_progress".format(task_id=task_id))
            percent = lines[-1]
            CommonConfig.default_logger.info("percent=" + str(percent))

            if percent == "1.00":
                status = True
                executeStatus = "SUCCESS"
                errorCode = 0
                errorMsg = ""
            elif pid_exists:
                status = True
                executeStatus = "RUNNING"
                errorCode = 0
                errorMsg = ""
            else:
                raise MorseException(result_code.CHECK_PROGRESS_ERROR)

        elif taskType == "predict":

            with open(os.path.join(absolute_path,
                                   'tfe/{task_id}/predict_pid'.format(task_id=task_id)), 'r') as f:
                pid = f.readline()
            if pid is None or pid == "" or pid == "None":
                raise MorseException(result_code.FILE_IS_EMPTY_ERROR,
                                     filename='tfe/{task_id}/predict_pid'.format(task_id=task_id))
            pid = int(pid)
            pid_exists = check_pid(pid)

            with open(os.path.join(absolute_path,
                                   "tfe/{task_id}/predict_progress".format(task_id=task_id)), "r") as f:
                lines = f.readlines()
            if lines is None or lines == "" or pid == "None":
                raise MorseException(result_code.FILE_IS_EMPTY_ERROR,
                                     filename="tfe/{task_id}/predict_progress".format(task_id=task_id))
            percent = lines[-1]

            if percent == "1.00":
                status = True
                executeStatus = "SUCCESS"
                errorCode = 0
                errorMsg = ""
            elif pid_exists:
                status = True
                executeStatus = "RUNNING"
                errorCode = 0
                errorMsg = ""
            else:
                raise MorseException(result_code.CHECK_PROGRESS_ERROR)

        # else:
            # assert taskType == "train_and_predict"
        elif taskType == "train_and_predict":
            assert taskType == "train_and_predict", "error taskType:{}".format(taskType)

            with open(os.path.join(absolute_path,
                                   'tfe/{task_id}/train_and_predict_pid'.format(task_id=task_id)), 'r') as f:
                pid = f.readline()
            if pid is None or pid == "" or pid == "None":
                raise MorseException(result_code.FILE_IS_EMPTY_ERROR,
                                     filename='tfe/{task_id}/train_and_predict_pid'.format(task_id=task_id))
            pid = int(pid)
            CommonConfig.default_logger.info("pid=" + str(pid))

            pid_exists = check_pid(pid)
            # --------------train progress----------------------------------------
            trian_progress_file = os.path.join(absolute_path, "tfe/" + task_id + "/train_progress")
            if not os.path.exists(trian_progress_file):
                CommonConfig.error_logger.exception(
                    'trian_progress_file {} does not exists'.format(trian_progress_file))
                raise MorseException(result_code.FILE_NOT_EXIST_ERROR, filename=trian_progress_file)
            with open(trian_progress_file, "r") as f:
                lines = f.readlines()
            if lines is None or lines == "":
                raise MorseException(result_code.FILE_IS_EMPTY_ERROR,
                                     filename=trian_progress_file)
            percent_train = lines[-1]

            CommonConfig.default_logger.info(
                "percent_train=" + str(percent_train))

            # --------------predict progress---------------------------------

            predict_progress_file = os.path.join(absolute_path, "tfe/" + task_id + "/predict_progress")
            if not os.path.exists(predict_progress_file):
                CommonConfig.error_logger.exception(
                    'predict_progress_file {} does not exists'.format(predict_progress_file))
                raise MorseException(result_code.FILE_NOT_EXIST_ERROR, filename=predict_progress_file)
            with open(predict_progress_file, "r") as f:
                percent_predict = f.readlines()
            if percent_predict is None or percent_predict == "":
                raise MorseException(result_code.FILE_IS_EMPTY_ERROR, filename=predict_progress_file)
            CommonConfig.default_logger.info(
                "percent_predict=" + str(percent_predict))
            percent_predict = percent_predict[-1]

            if percent_predict == "1.00":
                status = True
                executeStatus = "SUCCESS"
                errorCode = 0
                errorMsg = ""
            elif pid_exists:
                status = True
                executeStatus = "RUNNING"
                errorCode = 0
                errorMsg = ""

            else:
                raise MorseException(result_code.CHECK_PROGRESS_ERROR)

            if percent_predict == "1.00":
                percent = "1.00"
            else:
                percent = str(float(percent_train) * 0.95 + float(percent_predict) * 0.05)

        else:
            raise MorseException(result_code.PARAM_ERROR, param='taskType')

    except MorseException as e:

        status = False
        errorMsg = e.get_message()
        errorCode = e.get_code()
        executeStatus = "FAILED"

    except Exception as e:

        status = False
        errorMsg = result_code.CHECK_PROGRESS_ERROR.get_message()
        errorCode = result_code.CHECK_PROGRESS_ERROR.get_code()
        executeStatus = "FAILED"
        CommonConfig.error_logger.exception(
            'check_progress error, exception msg:{}'.format(str(e)))

    percent = int(float(percent) * 100)
    return json.dumps({"status": status, "executeStatus": executeStatus,

                       "errorCode": errorCode, "errorMsg": errorMsg, "percent": percent})


@tfe_keeper.route('/kill_server', methods=['GET', 'POST'])
def kill_server():
    """
    iuput: taskId
    :return:
    status, 
    errorCode, 
    errorMsg
    """
    # print("kill_server")
    try:
        CommonConfig.default_logger.info("kill_server request:" + str(request))
        request_params = request.json
        CommonConfig.default_logger.info("kill_server request_params:" + str(request_params))
        task_id = request_params.get('taskId')
        CommonConfig.default_logger.info("task_id:" + str(task_id))

        # with open(os.path.join(absolute_path, 'tfe/{task_id}/
        # server_pid'.format(task_id=task_id)), 'r') as f:
        if not os.path.exists(os.path.join(absolute_path, 'tfe/server_pid')):
            status = True
            errorCode = 0
            CommonConfig.default_logger.info("file server_pid does not exists.")
            return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": ""})
        with open(os.path.join(absolute_path, 'tfe/server_pid'), 'r') as f:
            pid = f.readline()
        if pid is None:
            CommonConfig.default_logger.info("no need kill pid")
            return json.dumps({"status": True, "errorCode": 0, "errorMsg": ""})
        pid = int(pid)
        try:
            os.kill(pid, 9)
            # errorMsg = "killed {pid}".format(pid=pid)
            CommonConfig.default_logger.info("kill pid:{}".format(pid))
        except Exception as e:
            CommonConfig.default_logger.info("pid:{} is not alive".format(pid))
        status = True
        errorCode = 0
        # print("p.exitcode:", p.exitcode)

        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": ""})
    except Exception as e:
        CommonConfig.error_logger.exception(
            'kill_server error, exception msg:{}'.format(str(e)))

        status = False
        errorCode = result_code.KILL_SERVER_ERROR.get_code()
        errorMsg = str(e)
        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})


app.register_blueprint(tfe_keeper, url_prefix='/tfe_keeper')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8081", debug=True)
    # print(platform.system())

    # print(absolute_path)
    # status=_start_server(task_id="qqq", XOwner_iphost="127.0.0.1:5677", 
    # YOwner_iphost="127.0.0.1:5678", RS_iphost="127.0.0.1:5679", Player="XOwner")
    # print(status)
