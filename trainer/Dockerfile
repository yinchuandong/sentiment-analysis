FROM continuumio/miniconda3 as base


RUN apt-get update
RUN apt-get -y install gcc

RUN conda install jupyter -y
RUN pip install jupyterlab==0.32.1
RUN pip install jupyterlab-launcher==0.10.5


FROM base as dev
ADD ./trainer/requirements.txt /trainer/requirements.txt
RUN pip install -r /trainer/requirements.txt

ADD ./rubixml/rubixml /rubixml/rubixml
ADD ./rubixml/setup.py /rubixml/setup.py
WORKDIR /rubixml
RUN pip install .



RUN mkdir /trainr
WORKDIR /trainer
ENTRYPOINT jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token="abcd"
