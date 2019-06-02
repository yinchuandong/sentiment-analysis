FROM yinchuandong/miniconda-torch:v1.0.0 as base


ADD ./trainer/requirements.txt /trainer/requirements.txt
RUN pip install -r /trainer/requirements.txt

ADD ./rubixml/rubixml /rubixml/rubixml
ADD ./rubixml/setup.py /rubixml/setup.py
WORKDIR /rubixml
RUN pip install .



RUN mkdir /trainr
WORKDIR /trainer
ENTRYPOINT jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token="abcd"
