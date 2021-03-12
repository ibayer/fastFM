Citing fastFM
=============

The library fastFM is an academic project. The time and resources spent
developing fastFM are therefore justified by the number of citations of
the software. If you publish scientific articles using fastFM, please
cite the following article (bibtex entry
[citation.bib](http://jmlr.org/papers/v17/15-355.bib)).

> Bayer, I. \"fastFM: A Library for Factorization Machines\" Journal of
> Machine Learning Research 17, pp. 1-5 (2016)

fastFM: A Library for Factorization Machines
============================================

[![image](https://travis-ci.org/ibayer/fastFM.svg?branch=master)](https://travis-ci.org/ibayer/fastFM)

[![image](https://img.shields.io/badge/platform-OSX%7CLinux-lightgrey.svg)](https://travis-ci.org/ibayer/fastFM)

[![image](https://img.shields.io/pypi/l/Django.svg)](https://travis-ci.org/ibayer/fastFM)

This repository allows you to use Factorization Machines in **Python**
(2.7 & 3.x) with the well known **scikit-learn API**. All performance
critical code has been written in C and wrapped with Cython. fastFM
provides stochastic gradient descent (SGD) and coordinate descent (CD)
optimization routines as well as Markov Chain Monte Carlo (MCMC) for
Bayesian inference. The solvers can be used for regression,
classification and ranking problems. Detailed usage instructions can be
found in the [online documentation](http://ibayer.github.io/fastFM) and
on [arXiv](http://arxiv.org/abs/1505.00641).

Supported Operating Systems
---------------------------

fastFM has a continuous integration / testing servers (Travis) for
**Linux (Ubuntu 14.04 LTS)** and **OS X Mavericks**. Other OSs are not
actively supported.

Usage
-----

``` {.python}
from fastFM import als
fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
fm.fit(X_train, y_train)
y_pred = fm.predict(X_test)
```

Tutorials and other information are available
[here](http://arxiv.org/abs/1505.00641). The C code is available as
[subrepository](https://github.com/ibayer/fastFM-core) and provides a
stand alone command line interface. If you still have **questions**
after reading the documentation please open an issue at GitHub.

+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+
\| Task \| Solver \| Loss \|
+================+==================+=============================+ \|
Regression \| als, mcmc, sgd \| Square Loss \|
+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+
\| Classification \| als, mcmc, sgd \| Probit(Map), Probit, Sigmoid\|
+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+
\| Ranking \| sgd \| BPR \|
+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+
*Supported solvers and tasks*

Installation
------------

**binary install (64bit only)**

`pip install fastFM`

**source install**

*Please make sure, that Python and OS bit version agree, e.g. 32bit
Python on 64bit OS won\'t work.*

``` {.bash}
# Install cblas and python-dev header (Linux only).
# - cblas can be installed with libatlas-base-dev or libopenblas-dev (Ubuntu)
$ sudo apt-get install python-dev libopenblas-dev

# Clone the repo including submodules (or clone + `git submodule update --init --recursive`)
$ git clone --recursive https://github.com/ibayer/fastFM.git

# Enter the root directory
$ cd fastFM

# Install Python dependencies (Cython>=0.22, numpy, pandas, scipy, scikit-learn)
$ pip install -r ./requirements.txt

# Compile the C extension.
$ make                      # build with default python version (python)
$ PYTHON=python3 make       # build with custom python version (python3)

# Install fastFM
$ pip install .
```

Tests
-----

The Python tests (`pip install nose`) can be run with:
`nosetests fastFM/fastFM/tests`

Please refer to the fastFM-core README for instruction on how to run the
C tests at `fastFM/fastFM-core/src/tests`.

Contribution
------------

-   Star this repository: keeps contributors motivated
-   Open an issue: report bugs or suggest improvements
-   Fix errors in the documentation: small changes matter
-   Contribute code

**Contributions are very welcome!** Since this project lives on GitHub
we recommend to open a pull request (PR) for code contributions as early
as possible. This is the fastest way to get feedback and allows [Travis
CI](https://travis-ci.org/ibayer/fastFM) to run checks on your changes.

Most information you need to setup your **development environment** can
be learned by adapting the great instructions on
<https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md>
. Please ensure that your contribution conforms to the
[PEP8](http://www.python.org/dev/peps/pep-0008/) Coding Style and
includes unit tests where appropriate. More valuable guidelines that
apply to fastFM can be found at
<http://scikit-learn.org/stable/developers/contributing.html#coding-guidelines>
.

**Contributors**

-   [aaossa](https://github.com/aaossa/)
-   [altimin](https://github.com/altimin)
-   [bdaskalov](https://github.com/bdaskalov)
-   [chezou](https://github.com/chezou)
-   [macks22](https://github.com/macks22)
-   [takuti](https://github.com/takuti)
-   [ibayer](https://github.com/ibayer)

License: BSD
------------

\<!\-- Matomo Image Tracker\--\> \<img
referrerpolicy=\"no-referrer-when-downgrade\"
src=\"<https://matomo.palaimon.io/matomo.php?idsite=2&rec=1>\"
style=\"border:0\" alt=\"\" /\> \<!\-- End Matomo \--\>
