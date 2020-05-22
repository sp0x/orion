from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

module = Extension("language_detector", [
    "language_detector.pyx",
    "std.cpp"
])

extensions = [module]
setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
