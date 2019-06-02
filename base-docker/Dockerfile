FROM continuumio/miniconda3


RUN apt-get update
RUN apt-get -y install gcc

RUN conda install jupyter -y
RUN pip install jupyterlab==0.32.1
RUN pip install jupyterlab-launcher==0.10.5


ADD ./base-docker/requirements.txt /base-docker/requirements.txt
RUN pip install -r /base-docker/requirements.txt
