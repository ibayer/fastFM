If you use this project please give credit by citing:

    Immanuel Bayer (2015): fastFM: A Library for Factorization Machines http://arxiv.org/abs/1505.00641



fastFM: A Library for Factorization Machines
============================================

.. image:: https://travis-ci.org/ibayer/fastFM.svg?branch=master
   :target: https://travis-ci.org/ibayer/fastFM
   
   
.. image:: https://img.shields.io/badge/platform-OSX|Linux-lightgrey.svg
  :target: https://travis-ci.org/ibayer/fastFM
  
.. image:: https://img.shields.io/pypi/l/Django.svg   
   :target: https://travis-ci.org/ibayer/fastFM

This repository allows you to use Factorization Machines in **Python** (2.7 & 3.x) with the well known **scikit-learn API**.
All performence critical code as been written in C and wrapped with Cython. fastFM provides
stochastic gradient descent (SGD) and coordinate descent (CD) optimization routines as well as Markov Chain Monte Carlo (MCMC) for Bayesian inference.
The solvers can be used for regression, classification and ranking problems. Detailed usage instructions can be found in the `online documentation  <http://ibayer.github.io/fastFM>`_ and on `arXiv <http://arxiv.org/abs/1505.00641>`_.

Supported Operating Systems
---------------------------
fastFM has a continous integration / testing servers (Travis) for **Linux (Ubuntu 14.04 LTS)**
and **OS X Mavericks**. Other OS are not actively supported.

Usage
-----
.. code-block:: python

    from fastFM import als
    fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
    fm.fit(X_train, y_train)
    y_pred = fm.predict(X_test)


Tutorials and other information are available `here <http://arxiv.org/abs/1505.00641>`_.
The C code is available as `subrepository <https://github.com/ibayer/fastFM-core>`_ and provides
a stand alone command line interface. If you have still **questions** after reading the documentation please open a issue at github.

+----------------+------------------+-----------------------------+
| Task           | Solver           | Loss                        |
+================+==================+=============================+
| Regression     | als, mcmc, sgd   | Square Loss                 |
+----------------+------------------+-----------------------------+
| Classification | als, mcmc, sgd   | Probit(Map), Probit, Sigmoid|
+----------------+------------------+-----------------------------+
| Ranking        | sgd              | BPR                         |
+----------------+------------------+-----------------------------+
*Supported solvers and tasks*

Installation
------------

**binary install (64bit only)**

``pip install fastFM``

**source install**

*Please make sure, that Python and OS bit version agree, e.g. 32bit Python on 64bit OS won't work.*

.. code-block:: bash

    # Install cblas and python-dev header (Linux only).
    # - cblas can be installed with libatlas-base-dev or libopenblas-dev (Ubuntu)
    $ sudo apt-get install python-dev libopenblas-dev

    # Clone the repro including submodules (or clone + `git submodule update --init --recursive`)
    $ git clone --recursive https://github.com/ibayer/fastFM.git

    # Enter the root directory
    $ cd fastFM

    # Install Python dependencies (Cython>=0.22, numpy, pandas, scipy, scikit-learn)
    $ pip install -r ./requirements.txt

    # Compile the C extension.
    $ make

    # Install fastFM
    $ pip install .


Tests
-----

The Python tests (``pip install nose``) can be run with:
``nosetests fastFM/fastFM/tests``

Please refere to the fastFM-core README for instruction on how to run the C tests at ``fastFM/fastFM-core/src/tests``.

Contribution
------------

* Star this repository: keeps contributors motivated
* Open a issue: report bugs or suggest improvements
* Fix errors in the documentation: small changes matter
* Contribute code

**Contributions are very wellcome!** Since this project lives on github we reommend
to open a pull request (PR) for code contributions as early as possible. This is the
fastest way to get feedback and allows `Travis CI <https://travis-ci.org/ibayer/fastFM>`_ to run checks on your changes.

Most information you need to setup your **development environment** can be learned by adapting the great instructions on https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md . Please ensure that your contribution conforms to the `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_ Coding Style and includes unit tests where appropriate. More valuable guidelines that apply to fastFM can be found at http://scikit-learn.org/stable/developers/contributing.html#coding-guidelines .


**Contributors**

* altimin
* bdaskalov
* chezou
* macks22
* takuti
* ibayer

License: BSD
------------
