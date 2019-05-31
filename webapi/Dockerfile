FROM tiangolo/uwsgi-nginx:python3.6 as base

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
