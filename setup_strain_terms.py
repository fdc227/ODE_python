from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

sourcefiles = ['strain_terms_cython.pyx', 'strain_terms_f.c']
extensions = [Extension("strain_terms_f_cython", 
                          sourcefiles,
                          include_dirs=[numpy.get_include()]
                          )]

setup(
  ext_modules = cythonize(extensions)
)