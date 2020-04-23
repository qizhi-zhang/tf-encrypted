# coding=utf-8
# qizhi.zqz
import numpy as np
import json
import mock
from unittest import TestCase, main
import tfe_keeper.train_lr as train_lr
import platform
import os

if platform.system() == "Darwin":
    absolute_path = "/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted"
else:
    absolute_path = "/app/file"

class TestBinning(TestCase):
    def setUp(self):

        conf = """
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
            "storagePath": "qqq/data/embed_op_fea_5w_format_x.csv",
            "fileRecord": 50038,
            "isContainY": False,
            "matchColNum": 2,
            "featureNum": 291
        },
        "node_id2": {
            "storagePath": "qqq/data/embed_op_fea_5w_format_y.csv",
            "fileRecord": 50038,
            "isContainY": True,
            "matchColNum": 2,
            "featureNum": 0
        }
    },
    "dataSetPredict": {
        "node_id1": {
            "storagePath": "qqq/data/embed_op_fea_5w_format_x.csv",
            "fileRecord": 50038,
            "isContainY": False,
            "matchColNum": 2,
            "featureNum": 291
        },
        "node_id2": {
            "storagePath": "qqq/data/embed_op_fea_5w_format_y.csv",
            "fileRecord": 50038,
            "isContainY": True,
            "matchColNum": 2,
            "featureNum": 0
        }
    }
}
        """
        conf = conf.replace("True", "true").replace("False", "false")
        # print(input)
        self.conf = json.loads(conf)
        print(conf)

    @mock.patch("tensorflow.Session.run")
    def test_run(self, run):
        run.return_value = np.array(["1, 0.1"] * 128)
        modelFilePath = os.path.join(absolute_path, "tfe/qqq/model")
        modelFilePlainTextPath = os.path.join("tfe/qqq/model/plaintext_model")
        train_lr.run(taskId="qqq", conf=self.conf, modelFileMachine="YOwner", modelFilePath=modelFilePath, modelFilePlainTextPath=modelFilePlainTextPath)


if __name__ == '__main__':
    main()
