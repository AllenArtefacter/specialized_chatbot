#!/usr/bin/env python

from setuptools import find_packages, setup
import os
from pathlib import Path

from setuptools import find_packages, setup

with open(os.path.join(Path(__file__).absolute().parents[0],"specialized_chatbot","VERSION")) as _f:
    __version__ = _f.read().strip()

setup(name='specialized_chatbot',
      version = __version__,
      description='Openai Chatbot for Customized Document Hub',
      packages=find_packages(),
      author='Allen',
      author_email='allen.liang@artefact.com',
      include_package_data=True,
      install_requires=[
        'llama-index>=0.4.28',
        'langchain>=0.0.112',
        'openai==0.27.2',
        'wikipedia',
        'unstructured',
        'PyPDF2',
        'docx2txt',
        'tensorflow>=2.0'
        ],
     )
