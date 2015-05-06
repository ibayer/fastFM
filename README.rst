A short paper describing the library is now available on 
arXiv http://arxiv.org/abs/1505.00641

This repository contains the python interface. Please have a look at https://github.com/ibayer/fastFM-core
if you are interessed in the command line interface or the solver source code (implemented in C).

GIT CLONE INSTRUCTION
=====================
This repository relays on sub-repositories just using ``git clone ..``
**doesn't fetch** them.

``git clone --recursive ..``

Or do the two-step dance if you wish.
You need to run ``git submodule update --init --recursive`` **from within** the
``fastFM-core/`` folder in order to clone them as well.


DEPENDENCIES
============

python libraries
----------------
* scikit-learn
* numpy
* scipy
* pandas
* cython

install with ``pip install -r /fastFM/requirements.txt``

C libraries
-----------
* CXSparse (included in submodule)
* glib-2.0
* gsl 1.15-1

This worked on ubuntu 14.04:
``sudo apt-get install libglib2.0-dev libatlas-base-dev python-dev``


Install fastFM (python)
=======================
first build the C libraries:
``(cd fastFM/; make)``

inplace for development:
``pip install -e fastFM/``


how to run tests / read examples
--------------------------------

pick your favorite test runner

``cd /fastFM/fastFM/tests/; py.test``
or 

``cd /fastFM/fastFM/tests/; nosetests``

Examples
--------
Please have a look add the files in `/fastFM/fastFM/tests/` for examples
on how to use FM for different tasks.
