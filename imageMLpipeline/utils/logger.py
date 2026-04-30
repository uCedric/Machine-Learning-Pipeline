import logging
import sys

def setup_logger(name=None):
    """
    Configures and returns a logger with a custom format: [Name] message
    """
    logger = logging.getLogger(name or "main")
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        # Custom format: [name] message
        # %(name)s will be replaced by the logger name (e.g., "ResNet18" or "main")
        formatter = logging.Formatter('[%(name)s] %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
    return logger

def get_logger(obj_or_name):
    """
    Returns a logger named after the class of the object or the provided string.
    Usage: self.logger = get_logger(self) or logger = get_logger(__name__)
    """
    if isinstance(obj_or_name, str):
        name = obj_or_name
    else:
        name = obj_or_name.__class__.__name__
    
    return setup_logger(name)
