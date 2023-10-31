from math import log
import pickle
import sys
import logging
from pathlib import Path
import yaml
import json
from loggers.log_factory import setup_logging

logging = setup_logging(__name__)


def model_predict(model, data):
    """
    Predicts the class of the given data using the provided model.

    Args:
        model (str or object): The path to the saved model or the model object itself.
        data (numpy array): The data to be classified.

    Returns:
        numpy array: The predicted class labels for the given data.
    """
    try:
        if isinstance(model, str):
            model = pickle.load(open(model, 'rb'))
        prediction = model.predict(data)
        logging.info(f"Successfully predicted class labels for data.")
        with open('src/features/defs.json', 'r') as json_file:
            class_mapping = json.load(json_file)
        reverse_mapping = {v: k for k, v in class_mapping.items()}
        prediction = [reverse_mapping[p] if p in reverse_mapping else "Unknown" for p in prediction] 
        return prediction
    except Exception as e:
        logging.error(f"Error occurred while predicting: {e}")
        return None

