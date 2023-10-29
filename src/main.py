import fire
from data.make_dataset import load_data
from src.preprocessing.data_cleaning import clean_data
from src.features.build_features import run_fe
from src.models.train_model import build_model




# f1 = "../data/raw/train_collision.csv"
# f2 = "../data/raw/train_main.csv"
# f3 = "../data/interim/train2.csv"

# col_to_bucket = ["Lumi"]
# cols_to_encode = ["Run", "nBJets", "Lumi"]

def run_pipeline(filepath1, filepath2, filepath3, col_to_bucket, cols_to_encode):
    """
    Runs the entire pipeline for the CERN classifier project.

    Args:
        filepath1 (str): Filepath for the first train dataset (collision).
        filepath2 (str): Filepath for the second train dataset (main).
        filepath3 (str): Filepath for where the merged data will be stored.
        col_to_bucket (list): List containing columns that need to be converted into buckets.
        cols_to_encode (list): List of column names to be one-hot encoded.
    """
    col_to_bucket = col_to_bucket.split(",")
    df = load_data([filepath1, filepath2], filepath3)
    df_cleaned = clean_data(df)
    df_fe = run_fe(df_cleaned, col_to_bucket, list(cols_to_encode))
    build_model(df_fe, "jets")

if __name__ == "__main__":
    fire.Fire(run_pipeline)