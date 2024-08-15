FROM nvcr.io/nvidia/pytorch:22.04-py3

RUN apt-get update -y
RUN apt-get install -y make
RUN apt-get install -y lzma
RUN apt-get install -y liblzma-dev
RUN apt-get install -y gcc 
RUN apt-get install -y zlib1g-dev bzip2 libbz2-dev
RUN apt-get install -y libreadline8 
RUN apt-get install -y libreadline-dev
RUN apt-get install -y sqlite3 libsqlite3-dev
RUN apt-get install -y openssl libssl-dev build-essential 
RUN apt-get install -y git curl wget
RUN apt-get install -y vim
RUN apt-get install -y sudo
RUN apt-get install -y libffi-dev
RUN apt-get install -y libgl1-mesa-dev

RUN apt-get install -y lsb-release gnupg
RUN apt-get install -y python3.8 python3-pip
RUN apt-get install -y byobu

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

ARG USERNAME
ENV USERNAME=$USERNAME
ARG UID
ENV UID=$UID
ARG GID
ENV GID=$GID
ARG ORIGINAL_DATA_LOC
ENV ORIGINAL_DATA_LOC=$ORIGINAL_DATA_LOC
ARG CONTAINER_HOMEDIR
ENV CONTAINER_HOMEDIR=$CONTAINER_HOMEDIR

RUN addgroup -gid $GID $USERNAME
RUN adduser $USERNAME --uid $UID --gid $GID
RUN usermod -aG sudo $USERNAME
RUN echo "$USERNAME:$USERNAME" | chpasswd

RUN mkdir -p $ORIGINAL_DATA_LOC
RUN chmod 777 $ORIGINAL_DATA_LOC
WORKDIR /home/$USERNAME
USER $USERNAME
