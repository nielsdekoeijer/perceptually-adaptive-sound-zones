FROM debian:buster

WORKDIR /workspace

# Install packages
ENV DEBIAN_FRONTEND=nonintercative
RUN apt-get -o Dpkg::Options::="--force-overwrite" install --fix-broken
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get update 
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    sudo \
    curl \
    libsndfile1-dev \
    software-properties-common \
    python3-dev \
    python3-distutils \
    python3-venv \
    python3

# Install python libs in venv
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN python3 get-pip.py
RUN rm get-pip.py
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install setuptools wheel
RUN python3 -m pip install numpy scipy matplotlib coloredlogs rir-generator mosek==9.3.21 pandas pesq soundfile pystoi
RUN python3 -m pip install soundfile sndfile
RUN python3 -m pip install pytest
RUN python3 -m pip install natsort
RUN python3 -m pip install joblib

# Note: this is our OLD version! Can be swapped out with libdetectability, TODO, though
ADD research-pydetectability /workspace/research-pydetectability
WORKDIR /workspace/research-pydetectability
RUN python3 -m pip install /workspace/research-pydetectability
RUN python3 -m pip install libdetectability

RUN apt-get install faketime -y
RUN python3 -m pip install qndiag

# You need a mosek.lic but obviously I'm not putting it on github :)
ADD mosek.lic /root/mosek/

WORKDIR /workspace
