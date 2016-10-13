from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension('ffm', ['fastFM/ffm.pyx'],
              libraries=['m', 'fastfm', 'cxsparse', 'blas'],
              library_dirs=['fastFM/', 'fastFM-core/bin/',
                            'fastFM-core/externals/CXSparse/Lib/'],
              include_dirs=['fastFM/', 'fastFM-core/include/',
                            'fastFM-core/externals/CXSparse/Include/',
              numpy.get_include()])]

setup(
    name='fastFM',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,

    packages=['fastFM'],

    version='0.2.6',
    url='http://ibayer.github.io/fastFM',
    author='Immanuel Bayer',
    author_email='immanuel.bayer@uni-konstanz.de',

    # Choose your license
    license='BSD',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'scikit-learn', 'scipy']
)
