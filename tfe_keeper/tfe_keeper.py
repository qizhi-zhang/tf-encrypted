from flask import Flask, redirect, url_for, request, Blueprint
import os
import json
import tensorflow as tf
from tf_encrypted.config import RemoteConfig
from multiprocessing import Process
import threading
from commonutils.common_config import CommonConfig
import train_lr
import predict_lr
import os
import platform
from commonutils.exception_utils.exception_utils import MorseException
from commonutils.exception_utils.result_code import result_code



absolute_path = None
if platform.system() == "Darwin":
    #os.putenv('absolute_path', "/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted")
    absolute_path="/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted"
else:
    #os.putenv('absolute_path', "/app")
    absolute_path="/app/file"


app = Flask(__name__)





@app.route('/service/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))



tfe_keeper = Blueprint('tfe_keeper', __name__)

@tfe_keeper.route('/detect_idle', methods=['GET', 'POST'])
def detect_idle():
    """

    :param ip_host:  xxx.xxx.xxx.xxx:host
    :return:   "idle" or "busy"
    """
    print("detect_idle start")
    try:
        #print("request:",request)
        CommonConfig.http_logger.info("detect_idle request:"+str(request))
        request_params = request.json
        CommonConfig.http_logger.info("detect_idle request_params:" + str(request_params))
        #print("request_params:",request_params)
        ip_host = request_params.get('ipHost')
        print("ip_host:", ip_host)
        status=_detect_idle(ip_host)
        print("status:", status)
        return json.dumps({"status": status})
    except Exception as e:
        CommonConfig.error_logger.exception(
            'detelt_idle error on input: {}, exception msg:{}'.format(str(request.json),str(e)))
        return e


def _detect_idle(ip_host):
    try:
        cluster = tf.train.ClusterSpec({"detect": [ip_host]})
        print("cluster:", cluster)
        server = tf.train.Server(cluster)
        print("server:", server)
        status="idle"
        #server.join()
    except Exception as e:
        #print(e)
        CommonConfig.error_logger.exception(
            '_detelt_idle error on input: {}, exception msg:{}'.format(str(ip_host), str(e)))
        status="busy"
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
    print("start_server")
    try:
        CommonConfig.http_logger.info("start_server request:" + str(request))
        request_params = request.json
        CommonConfig.http_logger.info("start_server request_params:" + str(request_params))
        task_id = request_params.get('taskId')
        print("task_id:", task_id)

        if task_id is None:
            raise MorseException(result_code.PARAM_ERROR, param="taskId")
        # if other is None:
        #     raise MorseException(result_code.PARAM_ERROR, param="other")

        RS_iphost = request_params.get('RS')
        if RS_iphost==None:
            RS_iphost=request_params.get("thirdOwner")
        print("RS_iphost:",RS_iphost)
        if RS_iphost==None:
            status = False
            errorCode = 1
            errorMsg = "nether RS nor thirdOwner are given"
            return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})

        XOwner_iphost = request_params.get('XOwner')
        if XOwner_iphost==None:
            XOwner_iphost = request_params.get('xOwner')
        print("XOwner_iphost=", XOwner_iphost)
        if XOwner_iphost==None:
            status = False
            errorCode = 1
            errorMsg = "nether xOwner nor XOwner are given"
            return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})


        YOwner_iphost = request_params.get('YOwner')
        if YOwner_iphost==None:
            YOwner_iphost = request_params.get('yOwner')
        if YOwner_iphost == None:
            status = False
            errorCode = 1
            errorMsg = "nether yOwner nor YOwner are given"
            return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})




        Player=request_params.get('Player')
        if Player==None:
            Player=request_params.get('player')
        if YOwner_iphost == None:
            status = False
            errorCode = 1
            errorMsg = "nether Player nor player are given"
            return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})

        if Player=="x_owner" or Player=="xOwner":
            Player="XOwner"
        if Player=="y_owner" or Player=="yOwner":
            Player="YOwner"
        if Player=="third_owner" or Player == 'thirdOwner':
            Player="RS"



        os.makedirs(os.path.join(absolute_path,"tfe/{task_id}".format(task_id=task_id)),exist_ok=True)
        p = Process(target=_start_server, args=(task_id, XOwner_iphost, YOwner_iphost, RS_iphost, Player))
        #status=_start_server(task_id, XOwner_iphost, YOwner_iphost, RS_iphost, Player)
        p.start()
        p.join(timeout=10)
        print("p.pid:")
        print(p.pid)
        if p.exitcode==0:


            #with open(os.path.join(absolute_path,'tfe/{task_id}/server_pid'.format(task_id=task_id)), 'w') as f:
            with open(os.path.join(absolute_path, 'tfe/server_pid'.format(task_id=task_id)), 'w') as f:
                f.write(str(p.pid))

            status=True
            errorCode=0
            errorMsg=""

        else:

            status = False
            errorCode = p.exitcode
            errorMsg = "start server over time=10s"
        # todo


        #print("p.exitcode:", p.exitcode)

        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})
    except MorseException as e:
        status = False
        errorMsg = e.get_message()
        errorCode = e.get_code()
        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})
    except Exception as e:
        err_msg = str(e)
        err_code = result_code.START_SERVER_SYSTEM_ERROR.get_code()

        #print(e)
        #return e
        CommonConfig.error_logger.exception(
            'start_server error , exception msg:{}'.format(str(e)))

        status = False
        errorCode = -1
        errorMsg = "start server faild"
        return json.dumps({"status": status, "errorCode": err_code, "errorMsg": err_msg})


def _start_server(task_id, XOwner_iphost, YOwner_iphost, RS_iphost, Player):
    try:
        config = """
{l}
    "XOwner": "{XOwner_iphost}",
    "YOwner": "{YOwner_iphost}",
    "RS": "{RS_iphost}"
{r}
        """.format(l="{", r="}", XOwner_iphost=XOwner_iphost, YOwner_iphost=YOwner_iphost, RS_iphost=RS_iphost)
        print("config:", config)
        os.system("pwd")
        print("absolute_path:",absolute_path)
        print("1. tfe config.json:", os.path.join(absolute_path, 'tfe/{task_id}/config.json'.format(task_id=task_id)))

        with open(os.path.join(absolute_path,'tfe/{task_id}/config.json'.format(task_id=task_id)), 'w') as f:
            f.write(config)

        print("2. tfe config.json:",os.path.join(absolute_path,'tfe/{task_id}/config.json'.format(task_id=task_id)))

        config = RemoteConfig.load(os.path.join(absolute_path,'tfe/{task_id}/config.json'.format(task_id=task_id)))
        server = config.server(Player, start=True)
        server.join()
    except Exception as e:
        CommonConfig.error_logger.exception(
            '_start_server error , exception msg:{}'.format(str(e)))
        #print(e)




@tfe_keeper.route('/train', methods=['GET', 'POST'])
def train():
    """
    input:
        taskId,algorithm,conf,modelFileMachine,modelFilePath
    :return:
        status,
        errorCode,
        errorMsg
    """


    print("train")
    try:
        CommonConfig.http_logger.info("train request:" + str(request))
        request_params = request.json
        CommonConfig.http_logger.info("train request_params:" + str(request_params))
        task_id = request_params.get('taskId')
        print("task_id:", task_id)
        algorithm = request_params.get('algorithm')
        modelFileMachine = request_params.get('modelFileMachine')
        if modelFileMachine=="x_owner" or modelFileMachine=="xOwner":
            modelFileMachine="XOwner"
        if modelFileMachine=="y_owner" or modelFileMachine=="yOwner":
            modelFileMachine="YOwner"
        if modelFileMachine=="third_owner" or modelFileMachine=="thirdOwner":
            modelFileMachine="RS"



        modelFilePath = request_params.get('modelFilePath')
        modelFilePath=os.path.join( absolute_path,modelFilePath)
        modelName = request_params.get('modelName')
        modelFilePlainTextPath = os.path.join(modelFilePath, modelName)
        conf=request_params.get('conf')

        test_flag=request_params.get('test_flag', False)

        if test_flag:
            tf_config_file=None
        else:
            tf_config_file =os.path.join(absolute_path,"tfe/{task_id}/config.json".format(task_id=task_id))

        # train_lr.run(task_id, conf, modelFileMachine, modelFilePath, tf_config_file=tf_config_file)

        #p = Process(target=train_lr.run, args=(task_id, conf, modelFileMachine, modelFilePath, modelFilePlainTextPath, tf_config_file))
        p = threading.Thread(target=train_lr.run, args=(task_id, conf, modelFileMachine, modelFilePath, modelFilePlainTextPath, tf_config_file))
        p.start()

        # CommonConfig.http_logger.info("train Process pid:" + str(p.pid))
        #
        # with open(os.path.join(absolute_path,'tfe/{task_id}/train_pid'.format(task_id=task_id)), 'w') as f:
        #     f.write(str(p.pid))

        CommonConfig.http_logger.info("train Process pid:" + str(p.name))

        with open(os.path.join(absolute_path, 'tfe/{task_id}/train_pid'.format(task_id=task_id)), 'w') as f:
            f.write(str(p.name))

        status=True
        errorCode=0
        errorMsg=""
        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})
    except Exception as e:
        #print(e)
        CommonConfig.error_logger.exception(
            'train error , exception msg:{}'.format(str(e)))
        #return e



@tfe_keeper.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    input:
        taskId,algorithm,conf,modelFileMachine,modelFilePath
    :return:
        status,
        errorCode,
        errorMsg
    """


    print("predict")
    try:
        CommonConfig.http_logger.info("pridict request:" + str(request))
        request_params = request.json
        CommonConfig.http_logger.info("predict request_params:" + str(request_params))
        task_id = request_params.get('taskId')
        print("task_id:", task_id)
        algorithm = request_params.get('algorithm')
        modelFileMachine = request_params.get('modelFileMachine')

        if modelFileMachine=="x_owner" or modelFileMachine=="xOwner":
            modelFileMachine="XOwner"
        if modelFileMachine=="y_owner" or modelFileMachine=="yOwner":
            modelFileMachine="YOwner"
        if modelFileMachine=="third_owner" or modelFileMachine=="thirdOwner":
            modelFileMachine="RS"

        modelFilePath = request_params.get('modelFilePath')
        modelFilePath = os.path.join(absolute_path, modelFilePath)
        conf=request_params.get('conf')
        test_flag = request_params.get('test_flag', False)

        progress_file = os.path.join(absolute_path,"tfe/" + task_id + "/predict_progress")

        if test_flag:
            tf_config_file=None
        else:
            tf_config_file =os.path.join(absolute_path,"tfe/{task_id}/config.json".format(task_id=task_id))

        #predict_lr.run(task_id, conf, modelFileMachine, modelFilePath, progress_file, tf_config_file)


        p = Process(target=predict_lr.run, args=(task_id, conf, modelFileMachine, modelFilePath, progress_file, tf_config_file))
        p.start()

        with open(os.path.join(absolute_path,'tfe/{task_id}/predict_pid'.format(task_id=task_id)), 'w') as f:
            f.write(str(p.pid))

        status=True
        errorCode=0
        errorMsg=""
        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg, "progressFile": progress_file})
    except Exception as e:
        #print(e)
        CommonConfig.error_logger.exception(
            'predict error , exception msg:{}'.format(str(e)))
        #return e





@tfe_keeper.route('/predict', methods=['GET', 'POST'])
def train_and_predict():
    """
    input:
        taskId,algorithm,conf,modelFileMachine,modelFilePath
    :return:
        status,
        errorCode,
        errorMsg
    """


    print("predict")
    try:
        CommonConfig.http_logger.info("pridict request:" + str(request))
        request_params = request.json
        CommonConfig.http_logger.info("predict request_params:" + str(request_params))
        task_id = request_params.get('taskId')
        print("task_id:", task_id)
        algorithm = request_params.get('algorithm')
        modelFileMachine = request_params.get('modelFileMachine')

        if modelFileMachine=="x_owner" or modelFileMachine=="xOwner":
            modelFileMachine="XOwner"
        if modelFileMachine=="y_owner" or modelFileMachine=="yOwner":
            modelFileMachine="YOwner"
        if modelFileMachine=="third_owner" or modelFileMachine=="thirdOwner":
            modelFileMachine="RS"

        modelFilePath = request_params.get('modelFilePath')
        modelFilePath = os.path.join(absolute_path, modelFilePath)
        modelName = request_params.get('modelName')
        modelFilePlainTextPath = os.path.join(modelFilePath, modelName)
        conf=request_params.get('conf')
        test_flag = request_params.get('test_flag', False)

        progress_file_predict = os.path.join(absolute_path,"tfe/" + task_id + "/predict_progress")

        if test_flag:
            tf_config_file=None
        else:
            tf_config_file =os.path.join(absolute_path,"tfe/{task_id}/config.json".format(task_id=task_id))

        #predict_lr.run(task_id, conf, modelFileMachine, modelFilePath, progress_file, tf_config_file)


        p = Process(target=predict_lr.run, args=(task_id, conf, modelFileMachine, modelFilePath, progress_file, tf_config_file))
        p.start()

        with open(os.path.join(absolute_path,'tfe/{task_id}/train_and_predict_pid'.format(task_id=task_id)), 'w') as f:
            f.write(str(p.pid))

        status=True
        errorCode=0
        errorMsg=""
        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg, "progressFile": progress_file})
    except Exception as e:
        #print(e)
        CommonConfig.error_logger.exception(
            'predict error , exception msg:{}'.format(str(e)))
        #return e





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

    print("check_progress")
    try:
        CommonConfig.http_logger.info("check_progress request:" + str(request))
        request_params = request.json
        CommonConfig.http_logger.info("check_progress request_params:" + str(request_params))
        task_id = request_params.get('taskId')
        print("task_id:", task_id)
        taskType = request_params.get('taskType')
        print("taskType:", taskType)

        percent = "0.00"
        if taskType=="train":
            try:
                with open(os.path.join(absolute_path,'tfe/{task_id}/train_pid'.format(task_id=task_id)), 'r') as f:
                    pid = f.readline()
                pid = int(pid)
                print("pid=",pid)

                pid_exists=check_pid(pid)

                with open(os.path.join(absolute_path,'tfe/{task_id}/train_progress'.format(task_id=task_id)), "r") as f:
                    percent = f.readlines()[-1]
                    print("percent=",percent)



                if percent=="1.00":
                    executeStatus = "SUCCESS"
                elif pid_exists:
                    executeStatus = "RUNNING"

                else:
                    executeStatus = "FAILED"



            except Exception as e:
                #print(e)
                CommonConfig.error_logger.exception(
                    'check_progress error , exception msg:{}'.format(str(e)))
                executeStatus="FAILED"


        else:
            assert taskType=="predict"
            try:
                with open(os.path.join(absolute_path,'tfe/{task_id}/predict_pid'.format(task_id=task_id)), 'r') as f:
                    pid = f.readline()
                pid = int(pid)
                pid_exists=check_pid(pid)

                with open(os.path.join(absolute_path,"tfe/{task_id}/predict_progress".format(task_id=task_id)), "r") as f:
                    percent = f.readlines()[-1]

                if percent == "1.00":
                    executeStatus = "SUCCESS"
                elif pid_exists:
                    executeStatus = "RUNNING"

                else:
                    executeStatus = "FAILED"



            except Exception as e:
                CommonConfig.error_logger.exception(
                    'check_progress error , exception msg:{}'.format(str(e)))
                executeStatus="FAILED"


        percent=int(float(percent)*100)
        status=True
        errorCode=0
        errorMsg=""
        return json.dumps({"status": status, "executeStatus": executeStatus, "errorCode": errorCode, "errorMsg": errorMsg, "percent": percent})
    except Exception as e:
        CommonConfig.error_logger.exception(
            'check_progress error , exception msg:{}'.format(str(e)))
        return e


@tfe_keeper.route('/kill_server', methods=['GET', 'POST'])
def kill_server():
    """
    iuput: taskId


    :return:
    status,
    errorCode,
    errorMsg
    """

    print("kill_server")
    try:
        CommonConfig.http_logger.info("kill_server request:" + str(request))
        request_params = request.json
        CommonConfig.http_logger.info("kill_server request_params:" + str(request_params))
        task_id = request_params.get('taskId')
        print("task_id:", task_id)

        #with open(os.path.join(absolute_path,'tfe/{task_id}/server_pid'.format(task_id=task_id)), 'r') as f:


        with open(os.path.join(absolute_path, 'tfe/server_pid'.format(task_id=task_id)), 'r') as f:
            pid=f.readline()


        pid=int(pid)
        os.kill(pid,9)
        errorMsg = "killed {pid}".format(pid=pid)




        status=True
        errorCode=0




        #print("p.exitcode:", p.exitcode)

        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})
    except Exception as e:
        CommonConfig.error_logger.exception(
            'kill_server error , exception msg:{}'.format(str(e)))

        status = True
        errorCode = 0
        errorMsg = "server is not running"
        return json.dumps({"status": status, "errorCode": errorCode, "errorMsg": errorMsg})


app.register_blueprint(tfe_keeper, url_prefix='/tfe_keeper')


if __name__ == '__main__':


    app.run(host="0.0.0.0",port="8080", debug = True)
    #print(platform.system())


    #print(absolute_path)
    #status=_start_server(task_id="qqq", XOwner_iphost="127.0.0.1:5677", YOwner_iphost="127.0.0.1:5678", RS_iphost="127.0.0.1:5679", Player="XOwner")
    #print(status)

