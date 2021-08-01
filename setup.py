import os
from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy as np

volume_ext = Extension('pyinverse.volume_new',
                       [os.path.join('pyinverse', 'volume_new.pyx')],
                       include_dirs=[np.get_include()],
                       language='c')

setup(name='pyinverse',
      version='0.1' ,
      description='Inverse problem tools for python.',
      author='Mark D. Butala',
      author_email='butala@illinois.edu',
      packages=['pyinverse'],
      ext_modules=cythonize([volume_ext], language_level=3),
      zip_safe=False
)
