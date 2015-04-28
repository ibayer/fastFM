from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension('ffm',
        ['fastFM/ffm.pyx'],
        libraries=['m', 'fastfm', 'cxsparse',# 'cblas',
            'gsl', 'gslcblas', #gsl-config --cflags --libs
            'glib-2.0'],
        library_dirs=['fastFM/', 'fastFM-core/bin/', 'fastFM-core/externals/CXSparse/Lib/',
            '/usr/lib/','/usr/lib/atlas-base/'],
        include_dirs=['fastFM/','fastFM-core/include/', 'fastFM-core/externals/CXSparse/Include/',
            '/usr/include/',
        '/usr/include/glib-2.0/', '/usr/lib/x86_64-linux-gnu/glib-2.0/include',
        numpy.get_include()])]
# pkg-config --cflags glib-2.0

setup(
    name = 'fastFM',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    packages=['fastFM'],
    version='0.1.0',
    author='Immanuel Bayer',
    author_email='immanuel.bayer@uni-konstanz.de'
)
