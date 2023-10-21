import logging
import yaml

import lime
import lime.lime_tabular
import pandas as pd


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


def run_lime_analysis(model, X_test: pd.DataFrame, y_test: pd.Series, instance_index: int):
    """
    Runs LIME analysis on the model for a specific instance in the test set and saves the plot to disk.
    """
    try:
        logging.info("Running LIME analysis.")

        # Create LIME explainer object
        explainer = lime.lime_tabular.LimeTabularExplainer(X_test.to_numpy(),
                                                           training_labels=y_test,
                                                           feature_names=X_test.columns,
                                                           class_names=['bijet', 'trijet', 'tetrajet'],
                                                           mode='classification')

        # Generate LIME explanation for a specific instance
        exp = explainer.explain_instance(X_test.iloc[instance_index].to_numpy(), model.predict_proba)

        # Save the LIME plot
        lime_plot_path = f"./lime_plot_{instance_index}.png"
        exp.save_to_file(lime_plot_path)

    except Exception as e:
        logging.error(f"An error occurred while running LIME analysis: {e}")