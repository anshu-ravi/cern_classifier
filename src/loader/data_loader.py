import logging
import os
from typing import Optional
import yaml
import pandas as pd
from loggers.log_factory import setup_logging

logging = setup_logging(__name__)


def data_loader(file_path: str) -> pd.DataFrame:
    """
    Loads data from a file located at the given file path. Supports CSV, XLSX, JSON, and TSV file formats.

    Args:
        file_path (str): The path to the file to load.

    Returns:
        pandas.DataFrame: The loaded data, or None if an error occurred.
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"Error: File path {file_path} does not exist")
            raise

        logging.info(f"Loading data from {file_path}")

        file_extension = os.path.splitext(file_path)[1]

        if file_extension == ".csv":
            data = pd.read_csv(file_path)
        elif file_extension == ".xlsx":
            data = pd.read_excel(file_path)
        elif file_extension == ".json":
            data = pd.read_json(file_path)
        elif file_extension == ".txt":
            data = pd.read_csv(file_path, sep="\t")
        else:
            logging.error(f"Error: Unsupported file type {file_extension}")
            raise
        logging.info(f"Successfully loaded data from {file_path}")
        return data

    except Exception as e:
        logging.error(f"Error: {e}")
        raise
