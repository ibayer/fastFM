FROM ubuntu:16.04

MAINTAINER Immanuel Bayer

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER root

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    build-essential cmake git

# Build fastfm-core

# Download and install miniconda.
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda2-4.3.27-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda

# Setup test virtual env

ARG TRAVIS_PYTHON_VERSION=3
ENV PY_VERSION=$TRAVIS_PYTHON_VERSION

RUN conda update -q conda && \
    conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION \
        cython numpy pandas scipy scikit-learn nose

#RUN echo 'source activate test-environment' > /tmp/activate_env.sh && \
#    /bin/bash /tmp/activate_env.sh && rm /tmp/activate_env.sh

#RUN [ “/bin/bash”, “-c”, “source activate test-environment && python setup.py develop” ]
#RUN ["/bin/bash", "-c", "source activate test-environment && conda info -a"]

#ADD fastFM-core2/ /fastfm/fastFM-core2/

# Build and install fastfm

# Run tests
