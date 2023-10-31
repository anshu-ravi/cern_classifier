import logging
import lime
import lime.lime_tabular
import pandas as pd
from loggers.log_factory import setup_logging


def run_lime_analysis(model, X_test: pd.DataFrame, y_test: pd.Series, instance_index: int):
    """
    Runs LIME analysis on the model for a specific instance in the test set and saves the plot to disk.
    """
    logging = setup_logging(__name__)
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