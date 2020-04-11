import os

with open('./tfe/server_pid', 'r') as f:
    pid = f.readline()

print(pid)