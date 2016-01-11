.. image:: https://travis-ci.org/ibayer/fastFM.svg
    :target: https://travis-ci.org/ibayer/fastFM
    
    
.. image:: https://img.shields.io/badge/platform-OSX|Linux-lightgrey.svg
    :target: https://travis-ci.org/ibayer/fastFM
    
.. image:: https://img.shields.io/pypi/l/Django.svg   
    :target: https://travis-ci.org/ibayer/fastFM
    
fastFM: A Library for Factorization Machines
============================================

This repository allows you to use the Factorization Machine model through the well known scikit-learn API.
All performence critical code as been written in C and wrapped with Cython. fastFM can be used for regression, classification and ranking problems. Detailed usage instructions can be found in the `online documentation  <http://ibayer.github.io/fastFM>`_ or on `arXiv <http://arxiv.org/abs/1505.00641>`_.

Usage
-----
.. code-block:: python

    from fastFM import als
    fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
    fm.fit(X_train, y_train)
    y_pred = fm.predict(X_test)


A short paper describing the library is now available on 
arXiv http://arxiv.org/abs/1505.00641

This repository contains the python interface. Please have a look at https://github.com/ibayer/fastFM-core
if you are interested in the command line interface or the solver source code (implemented in C).

GIT CLONE INSTRUCTION
=====================
This repository requires sub-repositories and just using ``git clone ..``
**doesn't fetch** them. Use
``git clone --recursive ..``
instead.

Otherwise you have to run ``git submodule update --init --recursive`` **from within** the
``fastFM-core/`` folder in order to get the sub-repositories.


DEPENDENCIES
============

python libraries
----------------
* scikit-learn
* numpy
* scipy
* pandas
* cython

Install
-------

install with ``pip install -r /fastFM/requirements.txt``

C libraries
-----------
* CXSparse (included as submodule)
* glib-2.0

This worked on ubuntu 14.04:
``sudo apt-get install libglib2.0-dev python-dev libatlas-base-dev``


Install fastFM (python)
=======================
**First build the C libraries:**
``(cd fastFM/; make)``

**For development install the lib inplace:**

(Run the following command from the same directory as ``git clone`` before.)

``pip install -e fastFM/``

Install on OSX
===============
Recommended way to manage dependencies is `Homebrew package manager
<https://brew.sh>`_. If you have brew installed, dependencies can be installed by running command ``brew install glib argp-standalone``. (Contributed by altimin)

Install on Windows
==================
It should be possible to compile the library on Windows.
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
