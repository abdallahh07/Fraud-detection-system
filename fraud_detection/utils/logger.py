import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.______)

    if not logger.handlers:  # avoids duplicate lines if called twice
        handler = logging.______()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger