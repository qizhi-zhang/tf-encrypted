from flask import Flask, redirect, url_for, request, Blueprint
import os
import json
import tensorflow as tf
from tf_encrypted.config import RemoteConfig
from multiprocessing import Process
import train_lr
import predict_lr
import os
import platform
absolute_path = None
if platform.system() == "Darwin":
    #os.putenv('absolute_path', "/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted")
    absolute_path="/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted"
else:
    #os.putenv('absolute_path', "/app")
    absolute_path="/app"


app = Flask(__name__)





@app.route('/success/<name>')
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
        print("request:",request)
        request_params = request.json
        print("request_params:",request_params)
        ip_host = request_params.get('ipHost')
        print("ip_host:", ip_host)
        state=_detect_idle(ip_host)
        print("state:", state)
        return json.dumps({"state": state})
    except Exception as e:
        return e


def _detect_idle(ip_host):
    try:
        cluster = tf.train.ClusterSpec({"detect": [ip_host]})
        print("cluster:", cluster)
        server = tf.train.Server(cluster)
        print("server:", server)
        state="idle"
        #server.join()
    except Exception as e:
        print(e)
        state="busy"
    return state





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
        print("request:",request)
        request_params = request.json
        print("request_params:",request_params)
        task_id = request_params.get('taskId')
        print("task_id:", task_id)

        RS_iphost = request_params.get('RS')
        if RS_iphost==None:
            RS_iphost=request_params.get("thirdOwner")
        print("RS_iphost:",RS_iphost)
        if RS_iphost==None:
            state = False
            errorCode = 1
            errorMsg = "nether RS nor thirdOwner are given"
            return json.dumps({"state": state, "errorCode": errorCode, "errorMsg": errorMsg})

        XOwner_iphost = request_params.get('XOwner')
        if XOwner_iphost==None:
            XOwner_iphost = request_params.get('xOwner')
        print("XOwner_iphost=", XOwner_iphost)
        if XOwner_iphost==None:
            state = False
            errorCode = 1
            errorMsg = "nether xOwner nor XOwner are given"
            return json.dumps({"state": state, "errorCode": errorCode, "errorMsg": errorMsg})


        YOwner_iphost = request_params.get('YOwner')
        if YOwner_iphost==None:
            YOwner_iphost = request_params.get('yOwner')
        if YOwner_iphost == None:
            state = False
            errorCode = 1
            errorMsg = "nether yOwner nor YOwner are given"
            return json.dumps({"state": state, "errorCode": errorCode, "errorMsg": errorMsg})




        Player=request_params.get('Player')
        if Player==None:
            Player=request_params.get('player')
        if YOwner_iphost == None:
            state = False
            errorCode = 1
            errorMsg = "nether Player nor player are given"
            return json.dumps({"state": state, "errorCode": errorCode, "errorMsg": errorMsg})

        if Player=="x_owner":
            Player="XOwner"
        if Player=="y_owner":
            Player="YOwner"
        if Player=="third_owner":
            Player="RS"



        os.makedirs(os.path.join(absolute_path,"file/{task_id}".format(task_id=task_id)),exist_ok=True)
        p = Process(target=_start_server, args=(task_id, XOwner_iphost, YOwner_iphost, RS_iphost, Player))
        #state=_start_server(task_id, XOwner_iphost, YOwner_iphost, RS_iphost, Player)
        p.start()
        #p.join(timeout=5)
        print("p.pid:")
        print(p.pid)

        with open(os.path.join(absolute_path,'file/{task_id}/server_pid'.format(task_id=task_id)), 'w') as f:
            f.write(str(p.pid))

        state=True
        errorCode=0
        errorMsg=""

        #print("p.exitcode:", p.exitcode)

        return json.dumps({"state": state, "errorCode": errorCode, "errorMsg": errorMsg})
    except Exception as e:
        print(e)
        return e


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
        print("1. tfe config.json:", os.path.join(absolute_path, 'file/{task_id}/config.json'.format(task_id=task_id)))

        with open(os.path.join(absolute_path,'file/{task_id}/config.json'.format(task_id=task_id)), 'w') as f:
            f.write(config)

        print("2. tfe config.json:",os.path.join(absolute_path,'file/{task_id}/config.json'.format(task_id=task_id)))

        config = RemoteConfig.load(os.path.join(absolute_path,'file/{task_id}/config.json'.format(task_id=task_id)))
        server = config.server(Player, start=True)
        server.join()
    except Exception as e:
        print(e)




@tfe_keeper.route('/train', methods=['GET', 'POST'])
def train():
    """
    input:
        taskId,algorithm,conf,modelFileMachine,modelFilePath
    :return:
        state,
        errorCode,
        errorMsg
    """


    print("train")
    try:
        print("request:",request)
        request_params = request.json
        print("request_params:",request_params)
        task_id = request_params.get('taskId')
        print("task_id:", task_id)
        algorithm = request_params.get('algorithm')
        modelFileMachine = request_params.get('modelFileMachine')
        if modelFileMachine=="x_owner":
            modelFileMachine="XOwner"
        if modelFileMachine=="y_owner":
            modelFileMachine="YOwner"
        if modelFileMachine=="third_owner":
            modelFileMachine="RS"



        modelFilePath = request_params.get('modelFilePath')
        modelFilePath=os.path.join( absolute_path,modelFilePath)
        conf=request_params.get('conf')

        test_flag=request_params.get('test_flag', False)

        if test_flag:
            tf_config_file=None
        else:
            tf_config_file =os.path.join(absolute_path,"file/{task_id}/config.json".format(task_id=task_id))

        # train_lr.run(task_id, conf, modelFileMachine, modelFilePath, tf_config_file=tf_config_file)
        p = Process(target=train_lr.run, args=(task_id, conf, modelFileMachine, modelFilePath, tf_config_file))
        p.start()

        with open(os.path.join(absolute_path,'file/{task_id}/train_pid'.format(task_id=task_id)), 'w') as f:
            f.write(str(p.pid))

        state=True
        errorCode=0
        errorMsg=""
        return json.dumps({"state": state, "errorCode": errorCode, "errorMsg": errorMsg})
    except Exception as e:
        print(e)
        return e



@tfe_keeper.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    input:
        taskId,algorithm,conf,modelFileMachine,modelFilePath
    :return:
        state,
        errorCode,
        errorMsg
    """


    print("predict")
    try:
        print("request:",request)
        request_params = request.json
        print("request_params:",request_params)
        task_id = request_params.get('taskId')
        print("task_id:", task_id)
        algorithm = request_params.get('algorithm')
        modelFileMachine = request_params.get('modelFileMachine')
        if modelFileMachine=="x_owner":
            modelFileMachine="XOwner"
        if modelFileMachine=="y_owner":
            modelFileMachine="YOwner"
        if modelFileMachine=="third_owner":
            modelFileMachine="RS"

        modelFilePath = request_params.get('modelFilePath')
        conf=request_params.get('conf')
        test_flag = request_params.get('test_flag', False)

        progress_file = os.path.join(absolute_path,"file/" + task_id + "/predict_progress")

        if test_flag:
            tf_config_file=None
        else:
            tf_config_file =os.path.join(absolute_path,"file/{task_id}/config.json".format(task_id=task_id))

        #predict_lr.run(task_id, conf, modelFileMachine, modelFilePath, progress_file, tf_config_file)



        p = Process(target=predict_lr.run, args=(task_id, conf, modelFileMachine, modelFilePath, progress_file, tf_config_file))
        p.start()

        with open(os.path.join(absolute_path,'file/{task_id}/predict_pid'.format(task_id=task_id)), 'w') as f:
            f.write(str(p.pid))

        state=True
        errorCode=0
        errorMsg=""
        return json.dumps({"state": state, "errorCode": errorCode, "errorMsg": errorMsg, "progressFile": progress_file})
    except Exception as e:
        print(e)
        return e






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

    print("predict")
    try:
        print("request:", request)
        request_params = request.json
        print("request_params:", request_params)
        task_id = request_params.get('taskId')
        print("task_id:", task_id)
        taskType = request_params.get('taskType')
        print("taskType:", taskType)

        percent = "0.00"
        if taskType=="train":
            try:
                with open(os.path.join(absolute_path,'file/{task_id}/train_pid'.format(task_id=task_id)), 'r') as f:
                    pid = f.readline()
                pid = int(pid)
                print("pid=",pid)

                pid_exists=check_pid(pid)

                with open(os.path.join(absolute_path,'file/{task_id}/train_progress'.format(task_id=task_id)), "r") as f:
                    percent = f.readlines()[-1]
                    print("percent=",percent)



                if percent=="1.00":
                    executeStatus = "SUCCESS"
                elif pid_exists:
                    executeStatus = "RUNNING"

                else:
                    executeStatus = "FAILED"



            except Exception as e:
                print(e)
                executeStatus="FAILED"


        else:
            assert taskType=="predict"
            try:
                with open(os.path.join(absolute_path,'file/{task_id}/predict_pid'.format(task_id=task_id)), 'r') as f:
                    pid = f.readline()
                pid = int(pid)
                pid_exists=check_pid(pid)

                with open(os.path.join(absolute_path,"file/{task_id}/predict_progress".format(task_id=task_id)), "r") as f:
                    percent = f.readlines()[-1]

                if percent == "1.00":
                    executeStatus = "SUCCESS"
                elif pid_exists:
                    executeStatus = "RUNNING"

                else:
                    executeStatus = "FAILED"



            except Exception as e:
                executeStatus="FAILED"


        percent=int(float(percent)*100)
        state=True
        errorCode=0
        errorMsg=""
        return json.dumps({"state": state, "executeStatus": executeStatus, "errorCode": errorCode, "errorMsg": errorMsg, "percent": percent})
    except Exception as e:
        print(e)
        return e


@tfe_keeper.route('/kill_server', methods=['GET', 'POST'])
def kill_server():
    """
    iuput: taskId


    :return:
    state,
    errorCode,
    errorMsg
    """

    print("kill_server")
    try:
        print("request:",request)
        request_params = request.json
        print("request_params:",request_params)
        task_id = request_params.get('taskId')
        print("task_id:", task_id)

        with open(os.path.join(absolute_path,'file/{task_id}/server_pid'.format(task_id=task_id)), 'r') as f:
            pid=f.readline()

        pid=int(pid)
        os.kill(pid,9)

        state=True
        errorCode=0
        errorMsg=""

        #print("p.exitcode:", p.exitcode)

        return json.dumps({"state": state, "errorCode": errorCode, "errorMsg": errorMsg})
    except Exception as e:
        print(e)
        return e





if __name__ == '__main__':

    app.register_blueprint(tfe_keeper, url_prefix='/tfe_keeper')
    app.run(host="0.0.0.0",port="8080", debug = True)
    #print(platform.system())


    #print(absolute_path)
    #state=_start_server(task_id="qqq", XOwner_iphost="127.0.0.1:5677", YOwner_iphost="127.0.0.1:5678", RS_iphost="127.0.0.1:5679", Player="XOwner")
    #print(state)

