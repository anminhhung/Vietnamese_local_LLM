FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore

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

# setup cuda
# RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# RUN pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install --no-cache-dir -r /workplace/requirements.txt
RUN pip3 install -U "ray[default]"

COPY . /workplace

WORKDIR /workplace