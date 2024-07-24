import logging
import sys
import os
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.log"
os.chdir('../')

LOG_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE_NAME)

os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE_NAME)

logging.basicConfig(filename=LOG_FILE_PATH, filemode='w',
                    level=logging.INFO,
                    format='[%(asctime)s] %(name)s  %(lineno)d - %(filename)s s- %(levelname)s %(message)s\n'
                    )


logging.info('logger started')
