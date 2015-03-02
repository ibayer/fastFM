GIT CLONE INSTRUCTION
=====================
git clone --recursive git@github.com:ibayer/fastFM.git

This repository relays on sub-repositories just using ``git clone ..``
doesn't fetch them.
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


Install fastFM (python)
=======================
first build the C libraries:
``(cd fastFM/; make)``

inplace for development:
``pip install -e fastFM/``


how to run tests
----------------

pick your favorite test runner

``cd /fastFM/fastFM/tests/; py.test``
or 

``cd /fastFM/fastFM/tests/; nosetests``
