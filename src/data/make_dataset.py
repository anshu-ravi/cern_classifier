# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from src.loader.data_loader import data_loader
from loggers.log_factory import setup_logging

def merge_data(df1: pd.DataFrame, df2: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Merges two dataframes on a given column name.

    Args:
        df1 (pandas.DataFrame): The first dataframe to merge.
        df2 (pandas.DataFrame): The second dataframe to merge.
        column_name (str): The name of the column to merge on.

    Returns:
        pandas.DataFrame: The merged dataframe.
    """
    logging = setup_logging(__name__)
    try:
        logging.info(f"Merging data on {column_name}")
        data = pd.merge(df1, df2, on=column_name)
        logging.info(f"Successfully merged data on {column_name}")
        return data

    except Exception as e:
        logging.error(f"Error merging data: {e}")
        raise


def load_data(input_filepath: List[str], output_filepath: str) -> pd.DataFrame:
    """
    Loads two dataframes from input filepaths, merges them on 'id' column and saves the merged dataframe to output filepath.

    Args:
        input_filepath (List[str]): List of two filepaths for input dataframes.
        output_filepath (str): Filepath to save the merged dataframe.

    Returns:
        None
    """
    logging = setup_logging(__name__)
    try:
        df1 = data_loader(input_filepath[0])
        df2 = data_loader(input_filepath[1])

        logging.info("Converting raw data into interim data")
        data = merge_data(df1, df2, "id")
        data.to_csv(output_filepath, index=False)
        logging.info(f"Successfully saved merged data to {output_filepath}")
        return data

    except Exception as e:
        logging.error(f"Error loading or merging data: {e}")
        raise
