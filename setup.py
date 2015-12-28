from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension('ffm',
        ['fastFM/ffm.pyx'],
        libraries=['m', 'fastfm', 'cxsparse', 'cblas', 'glib-2.0'],
        library_dirs=['fastFM/', 'fastFM-core/bin/', 'fastFM-core/externals/CXSparse/Lib/',
            '/usr/lib/','/usr/lib/atlas-base/'],
        include_dirs=['fastFM/','fastFM-core/include/', 'fastFM-core/externals/CXSparse/Include/',
            '/usr/include/',
        numpy.get_include()])]

setup(
    name = 'fastFM',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    packages=['fastFM'],
    version='0.1.1',
    author='Immanuel Bayer',
    author_email='immanuel.bayer@uni-konstanz.de'
)
