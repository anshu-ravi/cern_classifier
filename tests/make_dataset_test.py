import logging
import os
import tempfile
import pandas as pd
import pytest
from src.data.make_dataset import merge_data, load_data
from loggers.log_factory import setup_logging

logging = setup_logging(__name__)

@pytest.fixture
def sample_data():
    df1 = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })
    df2 = pd.DataFrame({
        "id": [1, 2, 4],
        "age": [25, 30, 35]
    })
    return df1, df2


def test_merge_data(sample_data):
    df1, df2 = sample_data
    merged = merge_data(df1, df2, "id")
    assert len(merged) == 2
    assert set(merged.columns) == {"id", "name", "age"}


def test_load_data(sample_data):
    df1, df2 = sample_data
    with tempfile.TemporaryDirectory() as tmpdir:
        input_filepaths = [
            os.path.join(tmpdir, "data1.csv"),
            os.path.join(tmpdir, "data2.csv")
        ]
        df1.to_csv(input_filepaths[0], index=False)
        df2.to_csv(input_filepaths[1], index=False)
        output_filepath = os.path.join(tmpdir, "merged.csv")
        load_data(input_filepaths, output_filepath)
        assert os.path.exists(output_filepath)
        merged = pd.read_csv(output_filepath)
        assert len(merged) == 2
        assert set(merged.columns) == {"id", "name", "age"}