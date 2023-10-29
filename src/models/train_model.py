import json
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from loggers.log_factory import setup_logging   

logging = setup_logging(__name__)

def data_split(df: pd.DataFrame, y_col: str, test_size: float) -> tuple:
    """
    Split the data into training and validation sets.

    Args:
        df (pandas.DataFrame): The input dataframe.
        y_col (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the validation split.

    Returns:
        tuple: A tuple containing the training and validation sets for X and y.

    Raises:
        ValueError: If the target column is not present in the dataframe.
        ValueError: If the test_size is not between 0 and 1.

    """
    if y_col not in df.columns:
        raise ValueError(
            f"Target column '{y_col}' not found in dataframe columns: {df.columns}"
        )

    if not 0 < test_size < 1:
        raise ValueError(f"Invalid test_size: {test_size}. Must be between 0 and 1.")

    X = df.drop(y_col, axis=1)
    y = df[y_col]

    try:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    except ValueError as e:
        logging.error(f"Error splitting data into train and validation sets: {e}")
        raise

    logging.info(
        f"Data split into train and validation sets. Train set size: {len(X_train)}. Validation set size: {len(X_valid)}"
    )

    return X_train, X_valid, y_train, y_valid


def scale_data(df: pd.DataFrame, test: bool = False) -> pd.DataFrame:
    """
    Scale the input data using StandardScaler.

    Args:
        df (pandas.DataFrame): The input dataframe.
        test (bool): A flag indicating whether the function is being called on test data.

    Returns:
        pandas.DataFrame: The scaled dataframe.

    Raises:
        TypeError: If the input dataframe is not a pandas dataframe.
        FileNotFoundError: If the scaler file is not found.
        ValueError: If the scaler file is empty.

    """
    try:
        if not test:
            scaler = StandardScaler()
            df = scaler.fit_transform(df)
            pickle.dump(scaler, open("./models/scaler.pkl", "wb"))
            logging.info("Training data scaled successfully.")
        else:
            try:
                scaler = pickle.load(open("./models/scaler.pkl", "rb"))
            except FileNotFoundError:
                logging.error("Scaler file not found.")
                raise
            if not scaler:
                logging.error("Scaler file is empty.")
                raise
            df = scaler.transform(df)
            logging.info("Test data scaled successfully.")
        return df
    except Exception as e:
        logging.error(f"Error scaling data: {e}")
        raise


def train_random_forest_model(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 2098
) -> RandomForestClassifier:
    """
    Train a random forest classifier on the input data.

    Args:
        X_train (pandas.DataFrame): The training data.
        y_train (pandas.Series): The training labels.
        random_state (int): The random state for reproducibility.

    Returns:
        RandomForestClassifier: The trained random forest classifier.

    Raises:
        TypeError: If the input data is not a pandas dataframe or series.

    """
    try:
        clf = RandomForestClassifier(
            n_estimators=500, criterion="entropy", random_state=random_state
        )
        clf.fit(X_train, y_train)
        logging.info("Random forest classifier trained successfully.")

        return clf

    except Exception as e:
        logging.error(f"Error training random forest classifier: {e}")
        raise


def evaluate_model(clf, X_valid, y_valid):
    """
    Evaluate the trained model on the validation set and save the classification report and metrics.

    Args:
        clf (sklearn.ensemble.RandomForestClassifier): The trained random forest classifier.
        X_valid (pandas.DataFrame): The validation data.
        y_valid (pandas.Series): The validation labels.

    Returns:
        None

    Raises:
        TypeError: If the input data is not a pandas dataframe or series.

    """
    try:
        preds = clf.predict(X_valid)
        clf_report = classification_report(y_valid, preds)
        logging.info(f"Classification report:\n{clf_report}")

        precision_macro = precision_score(preds, y_valid, average="macro")
        recall_macro = recall_score(preds, y_valid, average="macro")
        f1_macro = f1_score(preds, y_valid, average="macro")

        precision_micro = precision_score(preds, y_valid, average="micro")
        recall_micro = recall_score(preds, y_valid, average="micro")
        f1_micro = f1_score(preds, y_valid, average="micro")

        metrics = {
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
        }

        # Dump classification report to json file
        with open("../reports/classification_report.json", "w") as f:
            json.dump(clf_report, f)

        # Dump metrics to json file
        with open("../reports/metrics.json", "w") as f:
            json.dump(metrics, f)

        logging.info(
            "Model evaluated successfully. Classification report and metrics saved to reports folder."
        )

    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise


def improve_model(clf, X_train, y_train, param_grid, cv) -> GridSearchCV:
    """
    Use GridSearchCV to find the best hyperparameters for the model.

    Args:
        clf (sklearn.ensemble.RandomForestClassifier): The trained random forest classifier.
        X_train (pandas.DataFrame): The training data.
        y_train (pandas.Series): The training labels.
        param_grid (dict): The parameter grid to be used for GridSearchCV.
        cv (int): The number of cross-validation folds.

    Returns:
        None

    Raises:
        TypeError: If the input data is not a pandas dataframe or series.

    """
    try:
        if not param_grid:
            param_grid = {
                'n_estimators': [250, 500, 750, 1000],  # Number of trees in the forest
                'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
                'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
                'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
            }

        logging.info("Starting model improvement process...")
        clf = GridSearchCV(clf, param_grid, cv=cv, verbose=1, n_jobs=-1)
        clf.fit(X_train, y_train)

        logging.info(
            f"Best parameters found: {clf.best_params_} with accuracy: {clf.best_score_}"
        )
        logging.info("Model improvement process completed successfully. Storing model...")
        pickle.dump(clf, open("../models/model_after_hp.pkl", "wb"))
        return clf

    except Exception as e:
        logging.error(f"Error improving model: {e}")
        raise

def build_model(df: pd.DataFrame, y_col: str, test_size: float = 0.2, param_grid: dict = None, cv:int = 5):
    """
    Perform all model building steps and save the evaluation metrics in the reports folder

    Args:
    df (pd.DataFrame): The dataframe containing the data.
    y_col (str): The name of the target column.
    test_size (float): The proportion of the data to be used for validation.
    param_grid (dict): The hyperparameter grid to search over.
    cv (int): The number of cross-validation folds to use.

    Returns:
    clf: The trained classifier.
    """
    try:
        logging.info("Starting model building process...")
        X_train, X_valid, y_train, y_valid = data_split(df, y_col, test_size)
        X_train = scale_data(X_train)
        X_valid = scale_data(X_valid, test=True)
        clf = train_random_forest_model(X_train, y_train)
        evaluate_model(clf, X_valid, y_valid)
        logging.info("Model building process completed successfully.")
        clf = improve_model(clf, X_train, y_train, param_grid = param_grid, cv = cv)
        return clf
    except Exception as e:
        logging.error(f"Error: {e}")
