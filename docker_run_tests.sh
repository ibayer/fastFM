source activate test-environment
# Build fastFM-core
cd /fastfm/
make
pip install .
nosetests
