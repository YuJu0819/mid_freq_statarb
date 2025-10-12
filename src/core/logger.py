import logging, os

def get_logger(name: str, level: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    level_name = level or os.environ.get("LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    return logger
