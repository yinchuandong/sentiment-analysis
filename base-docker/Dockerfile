FROM continuumio/miniconda3


RUN apt-get update
RUN apt-get -y install gcc

RUN conda install jupyter -y
RUN pip install jupyterlab==0.32.1
RUN pip install jupyterlab-launcher==0.10.5

# Install CPU-version of torch
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp37-cp37m-linux_x86_64.whl
ADD ./base-docker/requirements.txt /base-docker/requirements.txt
RUN pip install -r /base-docker/requirements.txt
