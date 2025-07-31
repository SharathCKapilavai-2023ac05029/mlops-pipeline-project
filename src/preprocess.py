# src/preprocess.py

import pandas as pd
from sklearn.datasets import fetch_california_housing
from pathlib import Path
import logging, traceback
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mlflow.set_tracking_uri("http://127.0.0.1:5000")


def fetch_and_process_data():
    """
    Function that fetches the California Housing dataset from sklearn datasets, performs basic preprocessing,
    saves the processed data, creates and saves EDA plots, and logs them as artifacts to an MLflow run.
    """
    # Start an MLflow run
    with mlflow.start_run(run_name="data_preprocessing") as run:
        logging.info("Starting MLflow run for data preprocessing.")
        mlflow.set_tag("step", "preprocess")

        logging.info("Fetching California Housing dataset...")
        try:
            # Fetch the dataset
            cal_housing = fetch_california_housing(as_frame=True) #From "https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html"
            df = pd.concat([cal_housing.data, cal_housing.target], axis=1)

            # --- Save the processed data ---
            processed_data_dir = Path("data/processed")
            processed_data_dir.mkdir(parents=True, exist_ok=True)
            processed_data_path = processed_data_dir / "housing.csv"
            df.to_csv(processed_data_path, index=False)
            logging.info(f"Dataset successfully saved to {processed_data_path}")

            # --- Create and save EDA plots ---
            plots_dir = Path("data/plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # From the sklearn page, thought of plotting a couple of important graphs and eventually store in curren
            # mlflow runs storage
            # Plot 1: Distribution of Median House Value
            hist_plot_path = plots_dir / "housing_value_distribution.png"
            logging.info(f"Creating and saving histogram to {hist_plot_path}")
            plt.figure(figsize=(10, 6))
            sns.histplot(df['MedHouseVal'], kde=True, bins=30)
            plt.title('Distribution of Median House Value (in $100,000s)')
            plt.xlabel('Median House Value')
            plt.ylabel('Frequency')
            plt.savefig(hist_plot_path)
            plt.close()

            # Plot 2: Geographical Scatter Plot of House Values
            scatter_plot_path = plots_dir / "housing_geo_scatter.png"
            logging.info(f"Creating and saving scatter plot to {scatter_plot_path}")
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=df,
                x="Longitude",
                y="Latitude",
                size="Population",
                hue="MedHouseVal",
                palette="viridis",
                alpha=0.6,
                sizes=(20, 2000)
            )
            plt.title('Housing Value by Location and Population')
            plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(scatter_plot_path)
            plt.close()
            
            logging.info("Plots saved successfully.")

            # --- Log artifacts to MLflow ---
            logging.info("Logging artifacts to MLflow...")
            mlflow.log_artifact(processed_data_path, "processed_data")
            mlflow.log_artifacts(str(plots_dir), "plots") # Log the entire plots directory
            logging.info("Artifacts logged successfully.")

        except Exception as e:
            logging.error(f"An error occurred during data processing or plotting: {e}")
            logging.error("Exception occurred:\n" + traceback.format_exc())
            mlflow.end_run(status="FAILED") # Explicitly fail the run on error
            raise

if __name__ == "__main__":
    fetch_and_process_data()
