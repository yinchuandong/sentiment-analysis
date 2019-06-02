FROM tiangolo/uwsgi-nginx:python3.7 as base

# Install CPU-version of torch
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp37-cp37m-linux_x86_64.whl

# Install dependencies
ADD ./webapi/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Install rubixml
ADD ./rubixml/rubixml /rubixml/rubixml
ADD ./rubixml/setup.py /rubixml/setup.py
WORKDIR /rubixml
RUN pip install .

ADD ./webapi/app /app
ADD ./webapi/models /models
WORKDIR /app
