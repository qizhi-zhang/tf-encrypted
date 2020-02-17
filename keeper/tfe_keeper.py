from flask import Flask, redirect, url_for, request, Blueprint
import os
import json
import tensorflow as tf
from tf_encrypted.config import RemoteConfig
from multiprocessing import Process

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
        task_id = request_params.get('task_id')
        print("task_id:", task_id)
        RS_iphost = request_params.get('RS')
        print("RS_iphost:",RS_iphost)
        XOwner_iphost = request_params.get('XOwner')
        YOwner_iphost = request_params.get('YOwner')
        Player=request_params.get('Player')

        p = Process(target=_start_server, args=(task_id, XOwner_iphost, YOwner_iphost, RS_iphost, Player))
        #state=_start_server(task_id, XOwner_iphost, YOwner_iphost, RS_iphost, Player)
        p.start()
        #p.join(timeout=5)
        print("p.pid:")
        print(p.pid)
        with open('./{task_id}/pid'.format(task_id=task_id), 'w') as f:
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
        os.system("mkdir {task_id}".format(task_id=task_id))

        with open('./{task_id}/config.json'.format(task_id=task_id), 'w') as f:
            f.write(config)

        config = RemoteConfig.load('./{task_id}/config.json'.format(task_id=task_id))
        server = config.server(Player, start=True)
        server.join()
    except Exception as e:
        print(e)




@tfe_keeper.route('/kill_server', methods=['GET', 'POST'])
def kill_server():
    """

    :return:
    """

    print("kill_server")
    try:
        print("request:",request)
        request_params = request.json
        print("request_params:",request_params)
        task_id = request_params.get('task_id')
        print("task_id:", task_id)

        with open('./{task_id}/pid'.format(task_id=task_id), 'r') as f:
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
    app.run(debug = True)

    #state=_start_server(task_id="qqq", XOwner_iphost="127.0.0.1:5677", YOwner_iphost="127.0.0.1:5678", RS_iphost="127.0.0.1:5679", Player="XOwner")
    #print(state)

