from typing import Union, List
from pickle import dump, load
import json
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from loggers.log_factory import setup_logging

def create_buckets(
    data: pd.DataFrame, cols: List[str], test: bool = False):
    """
    Discretizes the values in the specified columns of the input DataFrame into
    four equal-frequency buckets (quartiles) using pandas.qcut, or into the
    specified buckets using pandas.cut.

    Args:
        data (pandas.DataFrame): The input DataFrame.
        cols (list of str): The names of the columns to discretize.
        test (bool, optional): Whether the function is being called on train
            data or test data. Defaults to False.

    Returns:
        pandas.DataFrame: The input DataFrame with the specified columns
        discretized into quartiles or the specified buckets.
    """
    logging = setup_logging(__name__)

    try:
        if not test:
            logging.info("Creating buckets for train data.")
            bb_bucket = None
            for col in cols:
                data[col], bb_bucket = pd.qcut(
                    data[col],
                    q=4,
                    labels=["Q1", "Q2", "Q3", "Q4"],
                    retbins=True,
                    precision=0,
                )
            if bb_bucket is not None:
                with open("./features/buckets.json", "w") as f:
                    json.dump(list(bb_bucket), f)
            return data
        else:
            with open("./features/buckets.json", "r") as f:
                buckets = json.load(f)
            for col in cols:
                logging.info("Assigning buckets for test")
                data[col] = pd.cut(
                    data[col],
                    bins=buckets,
                    labels=["Q1", "Q2", "Q3", "Q4"],
                    include_lowest=True)
            return data
    except Exception as e:
        logging.error(f"Error occurred while creating buckets: {e}")
        raise


def convert_to_categorical(data: pd.DataFrame, cols: List[str], test: bool):
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
    logging = setup_logging(__name__)

    try:
        if not test:
            logging.info("Converting columns to categorical for train data.")
            ohe_encoder = OneHotEncoder(sparse_output=False, drop="first")
            encoded_data = ohe_encoder.fit_transform(data[cols])
            dump(ohe_encoder, open("./features/encoder.pkl", "wb"))
        else:
            logging.info("Converting columns to categorical for test data.")
            ohe_encoder = load(open("./features/encoder.pkl", "rb"))
            encoded_data = ohe_encoder.transform(data[cols])

        column_names = ohe_encoder.get_feature_names_out(input_features=cols)
        encoded_df = pd.DataFrame(encoded_data, columns=column_names)
        data = data.drop(cols, axis=1)
        df = pd.concat([data, encoded_df], axis=1)

        return df
    except Exception as e:
        logging.error(f"Error occurred while converting columns to categorical: {e}")
        raise


def factorize(df: pd.DataFrame, test: bool) -> pd.DataFrame:
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
    logging = setup_logging(__name__)

    try:
        if not test:
            factor = pd.factorize(df["jets"])
            df["jets"] = factor[0]
            logging.info('Successfully factorized "jets" column.')
            defs = factor[1]
            with open("./features/defs.json", "w", encoding="utf-8") as json_file:
                json.dump(defs.tolist(), json_file)
            return df
        else:
            with open("./features/defs.json", encoding="utf-8") as json_file:
                defs = json.load(json_file)
            df["jets"] = pd.Categorical(df["jets"], categories=defs)
            df["jets"] = df["jets"].cat.codes
            logging.info('Successfully factorized "jets" column.')
            return df

    except Exception as e:
        logging.error(f"Error occurred while factorizing 'jets' column: {e}")
        raise

def run_fe(df: pd.DataFrame,
    bucket_cols: List[str],
    categorical_cols: List[str],
    test: bool = False):
    """
    Runs all feature engineering functions on the input DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        bucket_cols (list of str): The names of the columns to discretize.
        buckets (list or ndarray): The edges of the bins to use for
            discretization, as in pandas.cut.
        categorical_cols (list of str): The names of the columns to convert
            to categorical variables.
        test (bool, optional): Whether the function is being called on train
            data or test data. Defaults to False.

    Returns:
        pandas.DataFrame: The input DataFrame with all feature engineering
        functions applied.
    """
    logging = setup_logging(__name__)

    try:
        logging.info("Running feature engineering.")
        df = create_buckets(df, bucket_cols)
        df = convert_to_categorical(df, categorical_cols, test)
        df = factorize(df, test)
        logging.info("Feature engineering complete.")
        return df
    except Exception as e:
        logging.error(f"Error occurred while running feature engineering: {e}")
        raise