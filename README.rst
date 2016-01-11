If you use this project please give credit by citing:

    Immanuel Bayer (2014): fastFM: A Library for Factorization Machines http://arxiv.org/abs/1505.00641

.. image:: https://travis-ci.org/ibayer/fastFM.svg
    :target: https://travis-ci.org/ibayer/fastFM


.. image:: https://img.shields.io/badge/platform-OSX|Linux-lightgrey.svg
    :target: https://travis-ci.org/ibayer/fastFM

.. image:: https://img.shields.io/pypi/l/Django.svg   
    :target: https://travis-ci.org/ibayer/fastFM

fastFM: A Library for Factorization Machines
============================================

This repository allows you to use Factorization Machines in **Python** (2.7 & 3.5) with the well known **scikit-learn API**.
All performence critical code as been written in C and wrapped with Cython. fastFM provides
stochastic gradient descent (SGD) and coordinate descent (CD) optimization routines as well as Markov Chain Monte Carlo (MCMC) for Bayesian inference.
The solvers can be used for regression, classification and ranking problems. Detailed usage instructions can be found in the `online documentation  <http://ibayer.github.io/fastFM>`_ and on `arXiv <http://arxiv.org/abs/1505.00641>`_.

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

Installation
------------

.. code-block::

    # Install cblas and the python dev header (Linux only).
    $ sudo apt-get install python-dev libatlas-base-dev

    # Install all Python dependencies.
    $ pip install -r /fastFM/requirements.txt

    # Enter the root directory
    $ cd fastFM

    # Make sure the submodule are cloned if you didn't use ``git clone --recursive ..``
    $ git submodule update --init --recursive

    # Compile the C extension.
    $ make

    # Install fastFM
    pip install .


Tests
-----

The Python tests can be run with:
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
fastest way to get feedback and allows travis to run checks on your changes.

Most information you need to setup your **development environment** can be learned by adapting the great instructions on https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md . Please ensure that your contribution conforms to the `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_ Coding Style and includes unit tests where appropriate. More valuable guidelines that apply to fastFM can be found at http://scikit-learn.org/stable/developers/contributing.html#coding-guidelines .


**Contributors**

* chezou
* bdaskalov
* altimin
* takuti
* ibayer

License: BSD
------------
