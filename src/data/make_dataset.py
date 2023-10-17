# -*- coding: utf-8 -*-
import logging
import sys
from pathlib import Path
from typing import List
import pandas as pd
import yaml
import click
from dotenv import find_dotenv, load_dotenv

sys.path.append(
    str(
        Path(
            "/mnt/c/Users/ransh/Documents/IE University/Year 5/Sem1/MLOps/cern_classifer/"
        ).resolve()
    )
)
from src.loader.data_loader import data_loader


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
    try:
        logging.info(f"Merging data on {column_name}")
        data = pd.merge(df1, df2, on=column_name)
        logging.info(f"Successfully merged data on {column_name}")
        return data

    except Exception as e:
        logging.error(f"Error merging data: {e}")
        raise


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True), nargs=2)
@click.argument("output_filepath", type=click.Path())
def load_data(input_filepath: List[str], output_filepath: str) -> None:
    """
    Loads two dataframes from input filepaths, merges them on 'id' column and saves the merged dataframe to output filepath.

    Args:
        input_filepath (List[str]): List of two filepaths for input dataframes.
        output_filepath (str): Filepath to save the merged dataframe.

    Returns:
        None
    """
    try:
        df1 = data_loader(input_filepath[0])
        df2 = data_loader(input_filepath[1])

        data = merge_data(df1, df2, "id")
        data.to_csv(output_filepath)
        logging.info(f"Successfully saved merged data to {output_filepath}")

    except Exception as e:
        logging.error(f"Error loading or merging data: {e}")
        raise


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Converting raw data into interim data")
    load_data()


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
