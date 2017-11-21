#!/bin/bash
#
#Author: Likhith Chitneni
#License: BSD 3 Clause license - https://opensource.org/licenses/BSD-3-Clause 
#

set -e -x

# Install any system packages required here
#yum install -y $PACKAGE_TO_BE_INSTALLED

#Remove Python 2.6 and 3.3 since numpy requires >=2.7 or >=3.4
rm -rf /opt/python/cpython-2.6.9-*
rm -rf /opt/python/cp33-cp33m

#Make fastFM-core
cd /io/fastFM-core
make
cd /

#Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install -r /io/requirements.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install fastFM --no-index -f /io/wheelhouse
    "${PYBIN}/pip" install nose
    (cd "$HOME"; "${PYBIN}/nosetests" /io/fastFM/tests)
done

mv /io/wheelhouse /io/dist
