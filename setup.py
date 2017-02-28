from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize    


extensions = [Extension('magis.models.abstract.boundaries', ["magis/models/abstract/boundaries.pyx"]),
              Extension('magis.utils.cyutils', ["magis/utils/cyutils.pyx"])]

setup(
    name='magis',
    version='0.1dev',
    packages=['magis',],
    install_requires=['cython==0.17'],
    license='MIT',
    cmdclass={'build_ext': build_ext},
    ext_modules = cythonize(extensions),
    long_description=open('README.md').read(),
)

