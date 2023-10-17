import logging
from typing import Union, List
from pickle import dump, load
import json
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import yaml


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


def create_buckets(
    data: pd.DataFrame, cols: List[str], buckets: Union[None, List[float]] = None
):
    """
    Discretizes the values in the specified columns of the input DataFrame into
    four equal-frequency buckets (quartiles) using pandas.qcut, or into the
    specified buckets using pandas.cut.

    Args:
        data (pandas.DataFrame): The input DataFrame.
        cols (list of str): The names of the columns to discretize.
        buckets (list or ndarray, optional): The edges of the bins to use for
            discretization, as in pandas.cut. For train data, this should be
            None as the bins will be computed and for test data,
            this should be passed as the bins computed for the train data.

    Returns:
        pandas.DataFrame: The input DataFrame with the specified columns
        discretized into quartiles or the specified buckets.
        List[float]: The bins computed for the train data.
    """
    try:
        if buckets is None:
            logging.info("Creating buckets for train data.")
            for col in cols:
                data[col], buckets = pd.qcut(
                    data[col],
                    q=4,
                    labels=["Q1", "Q2", "Q3", "Q4"],
                    retbins=True,
                    precision=0,
                )
            return data, list(buckets) if buckets is not None else []

        for col in cols:
            logging.info("Assigning buckets for test")
            data[col] = pd.cut(
                data[col],
                bins=buckets,
                labels=["Q1", "Q2", "Q3", "Q4"],
                include_lowest=True,
            )
        return data
    except Exception as e:
        logging.error(f"Error occurred while creating buckets: {e}")
        raise


def convert_to_categorical(data: pd.DataFrame, cols: List[str], train: bool = True):
    """
    Converts the specified columns of the input DataFrame into one-hot encoded
    categorical variables using One-hot encoding.

    Args:
        data (pandas.DataFrame): The input DataFrame.
        cols (list of str): The names of the columns to convert.
        train (bool, optional): Whether the function is being called on train
            data or test data. Defaults to True.

    Returns:
        pandas.DataFrame: The input DataFrame with the specified columns
        converted to one-hot encoded categorical variables.
    """
    try:
        if train:
            logging.info("Converting columns to categorical for train data.")
            ohe_encoder = OneHotEncoder(sparse_output=False, drop="first")
            encoded_data = ohe_encoder.fit_transform(data[cols])
            dump(ohe_encoder, open("notebooks/encoder.pkl", "wb"))
        else:
            logging.info("Converting columns to categorical for test data.")
            ohe_encoder = load(open("notebooks/encoder.pkl", "rb"))
            encoded_data = ohe_encoder.transform(data[cols])

        column_names = ohe_encoder.get_feature_names_out(input_features=cols)
        encoded_df = pd.DataFrame(encoded_data, columns=column_names)
        data = data.drop(cols, axis=1)
        df = pd.concat([data, encoded_df], axis=1)

        return df
    except Exception as e:
        logging.error(f"Error occurred while converting columns to categorical: {e}")
        raise


def factorize(df: pd.DataFrame, test: bool = False) -> pd.DataFrame:
    """
    Factorizes the 'jets' column of the input DataFrame and saves the definitions
    to a JSON file if test is False. If test is True, the definitions are loaded
    from the JSON file and used to factorize the 'jets' column.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        test (bool, optional): Whether the function is being called on train
            data or test data. Defaults to False.

    Returns:
        pandas.DataFrame: The input DataFrame with the 'jets' column factorized.
    """
    try:
        if not test:
            factor = pd.factorize(df["jets"])
            df["jets"] = factor[0]
            logging.info('Successfully factorized "jets" column.')
            defs = factor[1]
            with open("./defs.json", "w", encoding="utf-8") as json_file:
                json.dump(defs.tolist(), json_file)
            return df
        else:
            with open("./defs.json", encoding="utf-8") as json_file:
                defs = json.load(json_file)
            df["jets"] = pd.Categorical(df["jets"], categories=defs)
            df["jets"] = df["jets"].cat.codes
            logging.info('Successfully factorized "jets" column.')
            return df

    except Exception as e:
        logging.error(f"Error occurred while factorizing 'jets' column: {e}")
        raise
