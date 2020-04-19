#!/bin/bash

echo "--> starting smodel runtime..."

export PYTHONPATH=$PYTHONPATH:./:./commonutils:./tfe_keeper:/app:/app/tfe_keeper
printenv

echo "--> start tfe_keeper -->"
python /app/tfe_keeper/tfe_keeper.py
echo "-------->"



