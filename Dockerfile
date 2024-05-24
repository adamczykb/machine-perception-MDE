FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install -y python3 python3-pip wget parallel imagemagick ffmpeg libsm6 libxext6 

RUN pip3 install torch torchvision

COPY . /app
WORKDIR /app

RUN pip3 install -r requirements.txt

ENV PYTHONPATH=/app
ENV LC_ALL=C
RUN bash
