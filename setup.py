#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='RL Framework',
      version='1.0',
      description='Framework for Reinforcement Learning agents',
      packages=find_packages(),
      install_requires=[
          'matplotlib',
          'numpy',
          'datetime',
          'joblib',
          'neurolab',
          'cvxopt',
          'unittest2',
          'inspyred',
          'scipy'
      ]
      )
