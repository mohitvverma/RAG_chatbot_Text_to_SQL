import logging
import sys
import os
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.log"

path = os.getcwd()
os.chdir('../')
LOG_PATH = os.path.join(os.getcwd() , "logs", LOG_FILE_NAME)


if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH, exist_ok=True)
else:
    pass

LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE_NAME)

logging.basicConfig(filename=LOG_FILE_PATH,
                    level=logging.INFO,
                    format='[%(asctime)s] %(name)s  %(lineno)d - %(filename)s s- %(levelname)s %(message)s')


logging.info('efvev')