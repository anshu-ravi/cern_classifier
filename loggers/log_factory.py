"""
This module provides logging configuration for the project.
"""

import logging
# import sys
import os
from typing import Optional
# from pathlib import Path
import yaml


ABS_PATH = "/mnt/c/Users/ransh/Documents/IE University/Year 5/Sem1/MLOps/cern_classifer/"

# sys.path.append(str(Path(ABS_PATH).resolve()))

def setup_logging(name) -> Optional[logging.Logger]:
    """
    Set up logging configuration for the project.

    Args:
        name (str): The name of the logger.

    Returns:
        logger (logging.Logger): The configured logger.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        KeyError: If the configuration file is missing required keys.
    """
    try:
        with open('../config.yaml', 'r', encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError:
        print("Configuration file not found.")
        return None
    except KeyError:
        print("Configuration file is missing required keys.")
        return None
    
    file_formatter = logging.Formatter(config['file_logging']['format'])
    console_formatter = logging.Formatter(config['console_logging']['format'])

    # Setting different levels for different handlers
    file_handler = logging.FileHandler(ABS_PATH + config['file_logging']['file_location'])
    file_handler.setLevel(getattr(logging, config['file_logging']['level']))
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config['console_logging']['level']))
    console_handler.setFormatter(console_formatter)

    # Setting different formats for different handlers
    logger = logging.getLogger(name)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

    return logger  # Return the configured logger