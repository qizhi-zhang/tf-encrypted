#!/bin/bash

echo "--> starting smodel runtime..."

export PYTHONPATH=$PYTHONPATH:./:./commonutils:./tfe_keeper:/app:/app/tfe_keeper
printenv

echo "--> start uwsgi -->"
/usr/local/bin/uwsgi  --ini  /app/conf/tfe_uwsgi.ini
echo "-------->"



