A short paper describing the library is now available on 
arXiv http://arxiv.org/abs/1505.00641

This repository contains the python interface. Please have a look at https://github.com/ibayer/fastFM-core
if you are interessed in the command line interface or the solver source code (implemented in C).

GIT CLONE INSTRUCTION
=====================
This repository relays on sub-repositories just using ``git clone ..``
**doesn't fetch** them.

use: ``git clone --recursive https://github.com/ibayer/fastFM.git``

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
* CXSparse (included as submodule)
* glib-2.0

This worked on ubuntu 14.04:
``sudo apt-get install libglib2.0-dev libatlas-base-dev python-dev``


Install fastFM (python)
=======================
**First build the C libraries:**
``(cd fastFM/; make)``

**For development install the lib inplace:**

(Run the following command from the same directory as ``git clone`` before.)

``pip install -e fastFM/``

Install on Windows / OSX
========================
It should be possible to compile the library on Windows and OSX.
I'm developing on linux but have received multiple requests from people who
want to run this library on other platforms.
Please let me know about issues you ran into or how you manged to compile on
other platfroms (or just open a PR) so that we include this information here.

how to run tests
----------------

pick your favorite test runner

``cd /fastFM/fastFM/tests/; py.test``
or 

``cd /fastFM/fastFM/tests/; nosetests``

Examples
--------
Please have a look add the files in ``/fastFM/fastFM/tests/`` for examples
on how to use FMs for different tasks.
