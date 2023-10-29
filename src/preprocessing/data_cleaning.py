import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from loggers.log_factory import setup_logging

logging = setup_logging(__name__)


def fill_missing_values(df, test=False):
    """
    This function fills missing values in the 'E1' and 'MR' columns of the input dataframe.
    If test is False, it fits a SimpleImputer to the data and saves it to a pickle file.
    If test is True, it loads the SimpleImputer from the pickle file and transforms the data.

    Args:
        df (pandas.DataFrame): The input dataframe.
        test (bool): Whether the function is being called on test data or not.

    Returns:
        pandas.DataFrame: The input dataframe with missing values filled.
    """
    try:
        df["MR"] = np.log10(df["MR"])
        df["E1"] = np.log10(df["E1"])

        if not test:
            logging.info("Fitting SimpleImputer to data...")
            imp_median = SimpleImputer(missing_values=np.nan, strategy="mean")
            df[["E1", "MR"]] = imp_median.fit_transform(df[["E1", "MR"]])
            pickle.dump(imp_median, open("./preprocessing/median_imputer.pkl", "wb"))
            return df

        else:
            logging.info("Loading SimpleImputer from pickle file...")
            imp_median = pickle.load(open("./preprocessing/median_imputer.pkl", "rb"))
            logging.info("Transforming test data...")
            df[["E1", "MR"]] = imp_median.transform(df[["E1", "MR"]])
            return df

    except Exception as e:
        logging.exception(
            "Exception occurred in fill_missing_values function: {}".format(str(e))
        )


def clean_data(
    df: pd.DataFrame, drop_columns: list = None, test: bool = False
) -> pd.DataFrame:
    """
    This function performs data cleaning on the input DataFrame. It drops the specified columns (if any), fills in missing values, and drops duplicates.

    Args:
        df (pd.DataFrame): Input pandas DataFrame to be cleaned.
        drop_columns (list, optional): List of column names to drop. Defaults to None.
        test (bool, optional): Boolean indicating if the function is being used for testing. Defaults to False.

    Returns:
        pd.DataFrame: Cleaned pandas DataFrame.

    Raises:
        ValueError: If the input DataFrame is empty.

    """
    try:
        logging.info("Starting data cleaning...")
        if df.empty:
            logging.error("Input DataFrame is empty. Cannot perform data cleaning.")
            raise ValueError("Input DataFrame is empty. Cannot perform data cleaning.")

        df_cleaned = df
        if drop_columns:
            df_cleaned = df.drop(drop_columns, axis=1)

        df_cleaned = fill_missing_values(df_cleaned, test=test)

        # Drop duplicates
        df_cleaned = df_cleaned.drop_duplicates()
        logging.info("Data cleaning completed successfully.")
        return df_cleaned

    except Exception as e:
        logging.error(f"Error occurred during data cleaning: {e}")
        raise
