"""Private predict, submit from YOwner.
by qizhi.zqz """

import tf_encrypted as tfe
import json
#from common_private import  ModelOwner, LogisticRegression, XOwner, YOwner
from common_private import  LogisticRegression
from read_data_tf import get_data_xy, get_data_x, get_data_y, get_data_id_with_y, get_data_id_with_xy
import sys
import time
import math
import os
import platform

if platform.system()=="Darwin":
    absolute_path="/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted"
else:
    absolute_path="/app"

def run(taskId,conf,modelFileMachine,modelFilePath, progress_file, tf_config_file=None):
    trainParams=conf.get("trainParams")


    batch_size = int(trainParams.get("batchSize"))






    dataSet=conf.get("dataSet")
    node_id1=dataSet.get("node_id1")
    node_id2=dataSet.get("node_id2")

    print("node1_containY:",node_id1.get("isContainY"))

    if (node_id1.get("isContainY")==True):
        featureNumX = int(node_id2.get("featureNum"))
        matchColNumX = int(node_id2.get("matchColNum"))
        path_x= node_id2.get("storagePath")
        record_num=int(node_id2.get("fileRecord"))

        featureNumY = int(node_id1.get("featureNum"))
        matchColNumY = int(node_id1.get("matchColNum"))
        path_y= node_id1.get("storagePath")
    else:
        assert node_id2.get("isContainY")==True
        featureNumY = int(node_id2.get("featureNum"))
        matchColNumY = int(node_id2.get("matchColNum"))
        path_y= node_id2.get("storagePath")
        record_num=int(node_id2.get("fileRecord"))

        featureNumX = int(node_id1.get("featureNum"))
        matchColNumX = int(node_id1.get("matchColNum"))
        path_x= node_id1.get("storagePath")

    path_x = os.path.join(absolute_path, path_x)
    path_y=os.path.join(absolute_path, path_y)
    batch_num=int(math.ceil(1.0*record_num/batch_size))
    feature_num=featureNumX+featureNumY



    # if len(sys.argv) >= 2:
    #   # config file was specified
    #   config_file = sys.argv[1]
    if tf_config_file:
      config = tfe.RemoteConfig.load(tf_config_file)
    else:
      # default to using local config
      config = tfe.LocalConfig([
          'XOwner',
          'YOwner',
          'RS'
      ])
    tfe.set_config(config)
    players = ['XOwner', 'YOwner', 'RS']
    prot = tfe.protocol.SecureNN(*tfe.get_config().get_players(players))
    tfe.set_protocol(prot)
    session_target = sys.argv[2] if len(sys.argv) > 2 else None




    # tfe.set_protocol(tfe.protocol.Pond(
    #     tfe.get_config().get_player(data_owner_0.player_name),
    #     tfe.get_config().get_player(data_owner_1.player_name)
    # ))

    @tfe.local_computation("XOwner")
    def provide_test_data_x(path="/Users/qizhi.zqz/projects/TFE/tf-encrypted/examples/test_on_morse_datas/data/embed_op_fea_5w_format_x.csv"):
        x = get_data_x(batch_size, path, featureNum=featureNumX, matchColNum=matchColNumX, epoch=2, clip_by_value=3.0, skip_row_num=1)
        return x

    #@tfe.local_computation("YOwner")
    def provide_test_data_y(path="/Users/qizhi.zqz/projects/TFE/tf-encrypted/examples/test_on_morse_datas/data/embed_op_fea_5w_format_y.csv"):
        idx, y = get_data_id_with_y(batch_size, path, matchColNum=matchColNumX, epoch=2,  skip_row_num=1)
        return idx, y

    #@tfe.local_computation("YOwner")
    def provide_test_data_xy(path="/Users/qizhi.zqz/projects/TFE/tf-encrypted/examples/test_on_morse_datas/data/embed_op_fea_5w_format_y.csv"):
        idx, x, y = get_data_id_with_xy(batch_size, path, featureNum=featureNumY, matchColNum=matchColNumX, epoch=2, clip_by_value=3.0, skip_row_num=1)
        return idx, x, y

    if (featureNumY==0):
        x_test = provide_test_data_x(path_x)
        idx, y_test = provide_test_data_y(path_y)
        y_test=prot.define_private_input("YOwner", lambda : y_test)
    else:
        idx, x_test1, y_test=provide_test_data_xy(path_y)
        x_test1=prot.define_private_input("YOwner", lambda : x_test1)
        y_test=prot.define_private_input("YOwner", lambda : y_test)

        x_test0=provide_test_data_x(path_x)
        x_test=prot.concat([x_test0, x_test1],axis=1)




    print("x_test:", x_test)
    print("y_test:", y_test)





    model = LogisticRegression(feature_num,learning_rate=0.1)



    load_op = model.load(modelFilePath,modelFileMachine)

    with tfe.Session() as sess:

        sess.run(tfe.global_variables_initializer(),
               tag='init')
        start_time=time.time()

        print("Loading model...")
        sess.run(load_op)
        print("Load OK.")


        #model.fit(sess, x_train, y_train, train_batch_num)
        #model.get_KS(sess, x_test,y_test, batch_num)

        #progress_file = "./" + taskId + "/predict_progress"


        model.predict(sess, x_test, os.path.join(absolute_path, "file/{task_id}/predict".format(task_id=taskId)), batch_num, idx, progress_file)



        test_time=time.time()-start_time
        print("predict_time=", test_time)

        with open(progress_file, "w") as f:
            f.write("1.00")
            f.flush()




if __name__=='__main__':

    with open('./qqq/conf', 'r') as f:
        conf=f.read()
        print(conf)
    conf=conf.replace("True","true").replace("False","false")
    #print(input)
    conf=json.loads(conf)
    print(conf)

    run(taskId="qqq", algorithm="tfe_lr", conf=conf, modelFileMachine="YOwner", modelFilePath="./qqq/model")
