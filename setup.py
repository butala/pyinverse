from setuptools import setup, Extension


setup(ext_modules=[
          Extension(name='lassere',
                    sources=['pyinverse/lassere/lassere.c',
                             'pyinverse/lassere/util.c'],
                    extra_compile_args=['-O3'])]
)
