FROM nvidia/cuda:10.2-base-ubuntu18.04

LABEL maintainer="aleksanderhan@gmail.com"
LABEL version="1.0"
LABEL description="trader training Dockerfile"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR = /

RUN apt update && apt upgrade -y

ADD train.py .
ADD requirements.txt .

RUN apt install -y cmake ssh libopenmpi-dev zlib1g-dev python3 python3-dev python3-pip python3-opencv
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD ["train.py"]
ENTRYPOINT ["python3"]