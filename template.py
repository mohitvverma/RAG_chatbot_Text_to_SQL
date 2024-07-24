import os
import logging

logging.basicConfig(level=logging.INFO,
                    format= '%(asctime)s -%(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )

project_name = 'TypeSQL'

list_of_files = [
    f'{project_name}/__init__.py',
    f'{project_name}/__main__.py',
]