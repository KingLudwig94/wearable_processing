from setuptools import setup, find_packages

setup(
    name='brainteaser-preprocessing',
    version='0.2.0',
    author='Luca Cossu',
    author_email='cossu.luca@gmail.com',
    packages=['brainteaser_preprocessing', 'brainteaser_preprocessing.hrv', 'brainteaser_preprocessing.utils',
              'brainteaser_preprocessing.activity', 'brainteaser_preprocessing.spo2',
              'brainteaser_preprocessing.respiration'],
    scripts=['bin/pipeline.py'],
    # url='http://pypi.python.org/pypi/PackageName/',
    # license='LICENSE.txt',
    description='A package to preprocess data coming from Garmin Vivoactive 4 via web API',
    long_description=open('README.txt').read(),
    install_requires=[
        "matplotlib==3.5.2",
        "pandas==1.3.4",
        "pobm==1.1.1",
        "numpy==1.21.6",
        "hrv-analysis==1.0.4",
        "scipy==1.7.3",
        "antropy==0.1.4"
    ],
)
