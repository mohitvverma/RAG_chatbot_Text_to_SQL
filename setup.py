import os
from setuptools import setup, find_packages

__version__ = '0.1.0'

REPO_NAME = 'NL_to_SQL'
AUTHOR_NAME = 'Mohit Verma'
AUTHOR_EMAIL = 'mohitvvermaa@outlook.com'
URL = 'https://github.com/mohitvvermaa/NL_to_SQL'


setup(
    name='NL_to_SQL',
    version=__version__,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    project_urls={
        'Documentation': URL+'/issues'
    }
)