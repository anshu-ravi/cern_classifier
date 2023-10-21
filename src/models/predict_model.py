import pickle
import sys
import logging
from pathlib import Path
import yaml
import json

sys.path.append(
    str(
        Path(
            "/mnt/c/Users/ransh/Documents/IE University/Year 5/Sem1/MLOps/cern_classifer/"
        ).resolve()
    )
)

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

