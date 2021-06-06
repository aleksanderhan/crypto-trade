FROM nvidia/cuda:10.2-base-ubuntu18.04

LABEL maintainer="aleksanderhan@gmail.com"
LABEL version="1.0"
LABEL description="trader training Dockerfile"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR = /

RUN apt update && apt upgrade -y
RUN apt install -y software-properties-common
RUN add-apt-repository universe
RUN apt update 

RUN apt install -y cmake ssh libopenmpi-dev zlib1g-dev git python3.7-tk #python3-opencv 

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:/miniconda/condabin:/miniconda/bin

RUN conda install -c anaconda python=3.7
RUN conda update --all
RUN rm /usr/bin/python3
RUN ln -s python3.7 /usr/bin/python3

RUN conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

COPY ./ .

RUN python3 -m pip install -r requirements.txt

ENTRYPOINT ["python3"]