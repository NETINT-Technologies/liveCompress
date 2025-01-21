import logging
import sys
from typing import Optional

def setup_logging(name: str, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging for training script.
    
    Args:
        name: The name of the logger (typically __name__)
        level: The logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: Optional file path to save logs
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # If logger already has handlers and it's properly configured, return it
    if logger.hasHandlers():
        return logger
    
    # Set level
    logger.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(file_handler)

    # Prevent logging from propagating to the root logger
    logger.propagate = False

    return logger