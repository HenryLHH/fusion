from setuptools import setup, find_packages

setup(name='ssr',
      packages=["ssr"],
      include_package_data=True,
      version='1.0.0',
      install_requires=['gym', 'ray==2.0.0'])

