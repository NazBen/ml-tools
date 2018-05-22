# Always prefer setuptools over distutils

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ml-tools',
    version='0.0.1',
    description='Machine learning tools that I constantly use.',
    long_description=open('README.md').read(),
    author='Nazih BENOUMECHIARA',
    author_email = 'nazih.benoumechiara@gmail.com',
    license='MIT',
    keywords=['machine learning'],
    packages=['mltools'],
    install_requires=required
)