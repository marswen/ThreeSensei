import os
import json
import datetime
import logging
import logging.handlers


def is_json(text):
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def init_logger(task_id):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    log_dir = os.path.join(os.path.dirname(__file__), '../log')
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, task_id+'.log'),
        when='D', backupCount=180, encoding='utf-8')
    file_handler.setLevel(level=logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s\n")
    file_formatter = logging.Formatter("%(asctime)s\n%(message)s\n")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger
