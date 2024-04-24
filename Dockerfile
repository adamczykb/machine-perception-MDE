FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install -y python3 python3-pip wget parallel imagemagick

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

COPY . /app
WORKDIR /app

RUN pip3 install -r requirements.txt

ENV PYTHONPATH=/app
ENV LC_ALL=C
RUN bash
