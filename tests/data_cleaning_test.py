import pandas as pd
import numpy as np
import pickle
from unittest.mock import patch
from sklearn.impute import SimpleImputer
from src.preprocessing.data_cleaning import fill_missing_values, clean_data

def test_fill_missing_values():
    # Create a DataFrame with missing values
    df = pd.DataFrame({
        "E1": [1, np.nan, 3],
        "MR": [4, 5, np.nan],
        "Other": ["a", "b", "c"]
    })

    # Call the fill_missing_values() function
    df_filled = fill_missing_values(df)

    # Check that the missing values have been filled
    assert not df_filled.isnull().any().any()

def test_clean_data():
    # Create a DataFrame with missing values and duplicates
    df = pd.DataFrame({
        "E1": [1, np.nan, 1],
        "MR": [4, 5, 4],
        "Other": ["a", "b", "a"]
    })

    # Call the clean_data() function
    df_cleaned = clean_data(df)

    # Check that the missing values have been filled
    assert not df_cleaned.isnull().any().any()

    # Check that the duplicates have been dropped
    assert df_cleaned.duplicated().sum() == 0

@patch('pickle.load')
def test_fill_missing_values_with_test(mock_pickle_load):
    # Create a DataFrame with missing values
    df = pd.DataFrame({
        "E1": [1, np.nan, 3],
        "MR": [4, 5, np.nan],
        "Other": ["a", "b", "c"]
    })

    # Mock the SimpleImputer loaded from pickle file
    mock_pickle_load.return_value = SimpleImputer(missing_values=np.nan, strategy="mean")

    # Call the fill_missing_values() function with test=True
    df_filled = fill_missing_values(df, test=True)

    # Check that the missing values have been filled
    assert not df_filled.isnull().any().any()