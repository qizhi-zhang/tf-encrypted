#!/bin/bash

#nosetests
#nosetests tf_encrypted/test/*_test.py  --cover-package=tfe --with-xunit --all-modules --traverse-namespace --with-coverage
python -m coverage run -p unittest/runMain.py test

coverage combine
coverage report >> result.md
echo "cat ../result.md"
sed 's/TOTAL/MYRESULT/' result.md

#coverage xml -o report/cobertura.xml --omit='tf_encrypted/*,commonutils/*,tfe_keeper/common_private.py,tfe_keeper/train_lr.py,tfe_keeper/predict_lr.py,tfe_keeper/train_and_predict_lr.py,tfe_keeper/read_data_tf.py,tfe_keeper/test.py'
coverage xml -o report/cobertura.xml --omit='tf_encrypted/*,commonutils/*,tfe_keeper/common_private.py,tfe_keeper/test.py'
curl http://aivolvo-dev.cn-hangzhou-alipay-b.oss-cdn.aliyun-inc.com/citools/covclient -o covclient
chmod +x covclient
./covclient --COV_FILE=report/cobertura.xml
./covclient --onlyWaitCompass
