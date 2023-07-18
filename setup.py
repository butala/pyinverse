from setuptools import setup, Extension


setup(ext_modules=[
          Extension(name='lasserre',
                    sources=['pyinverse/lasserre/lasserre.c',
                             'pyinverse/lasserre/util.c'],
                    extra_compile_args=['-O3'])]
)
