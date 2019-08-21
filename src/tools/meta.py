import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
import os
from configparser import ConfigParser

# ==============================#
# Logging info
# ==============================#
log_file = os.path.join(__file__, '..', '..', '..', 'logs', 'application.log')
handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=2)
handler.setFormatter(Formatter('[%(asctime)s] :: %(levelname)s :: MODULE %(module)s :: lINE %(lineno)d :: %(message)s'))

# ==============================#
# Configuration files
# ==============================#
config = ConfigParser()
config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'project.conf')
print(config_file)
config.read(config_file)

environment = config.get('GLOB', 'environment')

log = logging.getLogger()
if environment == 'DEV':
    DEBUG = True
    handler.setLevel(logging.DEBUG)
    log.setLevel(logging.DEBUG)
else:
    DEBUG = False
    handler.setLevel(logging.INFO)
    log.setLevel(logging.INFO)

log.addHandler(handler)