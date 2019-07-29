from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

sourcefiles = ['T_others_cython.pyx',]
extensions = [Extension("T_others_f_cython",
                          sourcefiles,
                          library_dirs=['~/Desktop/ODE_python/T_others_f.so'],
                          include_dirs=[numpy.get_include()]
                          )]

setup(
  ext_modules = cythonize(extensions)
)