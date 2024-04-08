import coloredlogs, logging 

def get_logger(name: str):
    """Creates a logger.
    
    Keyword arguments:
    name -- name of created logger
    """

    logging.basicConfig(filename='latest.log')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    coloredlogs.install(level='INFO')
    
    return logger
