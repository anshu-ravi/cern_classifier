import logging
import os
from typing import Union
import yaml
import pandas as pd

try:
    with open("config.yaml", "r", encoding="utf-8") as config_file:
        config_data = yaml.load(config_file, Loader=yaml.FullLoader)
        logging_rules = config_data.get("logging", {})
        logging.basicConfig(
            level=logging.getLevelName(config_data["logging"]["level"]),
            format=config_data["logging"]["format"],
        )

except FileNotFoundError:
    print(f"Error: File path not found.")
except yaml.YAMLError as e:
    print(f"Error parsing YAML: {e}")


def data_loader(file_path: str) -> Union[pd.DataFrame, None]:
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
            return None

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
            return None
        logging.info(f"Successfully loaded data from {file_path}")
        return data

    except Exception as e:
        logging.error(f"Error: {e}")
        return None
