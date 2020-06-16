#FROM python:3.4-alpine                 # << 基础镜像
#FROM continuumio/miniconda3
FROM registry.cn-hangzhou.aliyuncs.com/dtunion/morsetfe:vs_stf
ADD .  /TFE
# << 将当前目录复制到镜像中的/code/

WORKDIR /TFE
# << 设置工作目录

#RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
#RUN pip install sympy -i https://mirrors.aliyun.com/pypi/simple/
#RUN apt-get install wondershaper -y
#RUN wondershaper eth0 4096 4096
#RUN export PYTHONPATH=$PYTHONPATH:/morse-stf
ENV PYTHONPATH /TFE
#安装依赖

#EXPOSE 8887 8888 8889
# CMD ["python", "app.py"]
#ENTRYPOINT ["/bin/bash"]
#<< 设置默认启动命令