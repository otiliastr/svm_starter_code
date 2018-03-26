import os
from setuptools import setup, find_packages


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()

setup(
    name='hw4',
    version='0.1.dev0',
    license='Apache License 2.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    description='Kernel SVMs',
    url='https://github.com/otiliastr/svm_starter_code',
    author='Otilia Stretcu',
      author_email='ostretcu@cs.cmu.edu',
    install_requires=[
        'numpy', 'scipy']
)