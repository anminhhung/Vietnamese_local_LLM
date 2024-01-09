FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN apt-get install -y python3.10
RUN apt-get install -y python3-pip nano 

COPY requirements.txt /workplace/

RUN pip3 install -U setuptools
RUN pip3 install wheel==0.40.0 && \
    pip3 install --no-build-isolation Cython==0.29.36 
 

RUN pip3 install --no-cache-dir -r /workplace/requirements.txt

COPY . /workplace

WORKDIR /workplace