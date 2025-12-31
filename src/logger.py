import logging
import sys
import os
from config import Config

def setup_logging(log_filename=None, level=logging.INFO):
    
    try:
        os.makedirs(Config.BASE_DIR, exist_ok=True)
    except Exception:
        pass

    log_file = log_filename or os.path.join(Config.BASE_DIR, "scraper.log")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(asctime)s - %(name)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )
