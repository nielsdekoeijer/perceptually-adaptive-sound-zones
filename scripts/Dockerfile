FROM debian:buster

WORKDIR /workspace

# Install packages
ENV DEBIAN_FRONTEND=nonintercative
RUN apt-get -o Dpkg::Options::="--force-overwrite" install --fix-broken
RUN apt-get update
RUN apt-get install -y software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
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
# RUN python3.10 -m pip install dpam tensorflow-gpu scikit-learn tqdm resampy librosa
# RUN python3.10 -m pip install tensorflow
# RUN python3.10 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
RUN python3 -m pip install joblib

# Build ViSQOL
# RUN apt-get install -y curl gnupg git
# RUN python3 -m pip install numpy
# RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
# RUN mv bazel-archive-keyring.gpg /usr/share/keyrings
# RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
# RUN apt-get update && sudo apt-get install -y bazel-5.1.0
# WORKDIR /workspace
# RUN git clone https://github.com/google/visqol.git 
# WORKDIR /workspace/visqol
# RUN bazel-5.1.0 build :visqol -c opt
# RUN cp /workspace/visqol/bazel-bin/visqol /workspace/visqol-bin
# RUN cp -r /workspace/visqol/model /workspace/model

ADD research-pydetectability /workspace/research-pydetectability
WORKDIR /workspace/research-pydetectability
RUN python3 -m pip install /workspace/research-pydetectability
RUN python3 -m pip install libdetectability

RUN apt-get install faketime -y
RUN python3 -m pip install qndiag
ADD mosek.lic /root/mosek/
WORKDIR /workspace
