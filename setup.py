from setuptools import setup, find_packages

setup(
    name='torchvista',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'ipython>=7.0.0'
    ],
    package_data={
        'mylib': ['templates/*.html'],
    },
)
