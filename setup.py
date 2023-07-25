#!/usr/bin/env python3

from setuptools import setup # type: ignore
import up_multi_goal_split


long_description=\
'''
 ============================================================
    MULTI_GOAL_SPLIT
 ============================================================

    up_multi_goal_split is a package that allows for various compilations related to multiple goals
'''

setup(name='up_multi_goal_split',
      version=up_multi_goal_split.__version__,
      description='Unified Planning Integration of Multi-goal Split',
      long_description='Integration of multi-goal split compilations into the Unified Planning Framework',      
      author='Technion Cognitive Robotics Lab',
      author_email='karpase@technion.ac.il',
      url='https://github.com/aiplan4eu/up-multi-goal-split',
      classifiers=['Development Status :: 3 - Alpha',
               'License :: OSI Approved :: Apache Software License',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering :: Artificial Intelligence'
               ],
      packages=['up_multi_goal_split'],
      install_requires=[],
      python_requires='>=3.7',
      license='APACHE'
)
