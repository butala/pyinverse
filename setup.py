from setuptools import setup, Extension


setup(name='pyinverse',
      version='0.1' ,
      description='Inverse problem tools for python.',
      author='Mark D. Butala',
      author_email='butala@illinois.edu',
      packages=['pyinverse'],
      ext_modules=[
          Extension(name='lassere',
                    sources=['./lassere/lassere.c',
                             './lassere/util.c'],
                    extra_compile_args=['-O3'])],
      zip_safe=False
)
