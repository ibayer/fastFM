use ``git submodule update --init --recursive`` insubmodule folder
if any files are missing


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
inplace for development:
``pip install -e fastFM/``


how to run tests
----------------

pick your favorite test runner

``cd /fastFM/tests/; py.test``
or 

``cd /fastFM/tests/; nosetests``
