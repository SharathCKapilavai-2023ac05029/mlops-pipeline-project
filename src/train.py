# src/train.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml
import time
import joblib
import os, json

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Set the MLflow Tracking URI ---
# This tells MLflow to send data to the running UI server.
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
# Set the experiment name
mlflow.set_experiment("California Housing Prediction")
# Enable automatic system metrics logging
mlflow.enable_system_metrics_logging()


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, data_version):
    """Trains a model, evaluates it, and logs everything with MLflow."""
    with mlflow.start_run(run_name=model_name) as run:
        logging.info(f"Starting run for {model_name}")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("data_version", data_version)
        
        # Log model parameters
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())
        
        # Log dataset shape
        mlflow.log_param("training_rows", X_train.shape[0])
        mlflow.log_param("training_cols", X_train.shape[1])
        
        # Train the model and log training time
        logging.info("Training the model...")
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        logging.info(f"Training completed in {training_time:.2f} seconds.")
        
        # Make predictions
        logging.info("Making predictions...")
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        logging.info("Evaluating the model...")
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        logging.info(f"Metrics for {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        # Log metrics
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)
        
        # --- Create and log diagnostic plots ---
        plots_dir = Path("plots") / model_name
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Predictions vs. Actuals
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs. Predicted Values ({model_name})")
        plt.savefig(plots_dir / "actual_vs_predicted.png")
        plt.close()

        # Plot 2: Feature Importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X_train.columns
            forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            
            plt.figure(figsize=(10, 6))
            forest_importances.plot(kind='bar')
            plt.title(f"Feature Importances ({model_name})")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.savefig(plots_dir / "feature_importance.png")
            plt.close()
            
        # Log all plots in the directory
        mlflow.log_artifacts(str(plots_dir), artifact_path="plots")

        # --- Log the model with signature and input example ---
        logging.info("Logging the model with signature...")
        input_example = X_train.head()

        output_example = model.predict(input_example)
        signature = infer_signature(input_example, output_example)

        # mlflow.sklearn.log_model(
        #     sk_model=model, 
        #     artifact_path="model",
        #     signature=signature,
        #     input_example=input_example
        # )
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        # Save the model manually
        model_file = models_dir / f"{model_name}.joblib"
        joblib.dump(model, model_file)

        # Optionally log signature & input_example manually as artifacts

        # Save input example and signature (as JSON for inspection purposes)
        input_example_path = models_dir / f"{model_name}_input_example.json"
        signature_path = models_dir / f"{model_name}_signature.json"

        with open(input_example_path, "w") as f:
            json.dump(input_example.to_dict(orient="records"), f)

        with open(signature_path, "w") as f:
            json.dump(signature.to_dict(), f)

        # Log artifacts to MLflow
        mlflow.log_artifact(str(model_file))
        mlflow.log_artifact(str(input_example_path))
        mlflow.log_artifact(str(signature_path))
        
        # Log the training script itself
        mlflow.log_artifact(__file__, "code")
        
        logging.info(f"Run for {model_name} completed.")
        return run.info.run_id

def select_best_model(experiment_name, model_name):
    """Finds the best run in an experiment and registers its model."""
    logging.info(f"Promoting best model from experiment: '{experiment_name}'")
    client = MlflowClient()

    try:
        # Get the experiment ID from its name
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            logging.error(f"Experiment '{experiment_name}' not found.")
            return

        experiment_id = experiment.experiment_id

        # Search for the best run in the experiment based on the R2 score
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["metrics.test_r2 DESC"],
            max_results=1
        )

        if not runs:
            logging.error("No runs found in the experiment.")
            return

        best_run = runs[0]
        best_run_id = best_run.info.run_id

        r2_score = best_run.data.metrics.get("test_r2", "N/A")
        if isinstance(r2_score, float):
            logging.info(f"Best run found: {best_run_id} with R2 Score: {r2_score:.4f}")
        else:
            logging.info(f"Best run found: {best_run_id} with R2 Score: {r2_score}")

        model_uri = f"runs:/{best_run_id}/model"

        # Register the model from the best run
        logging.info(f"Registering model '{model_name}' from {model_uri}")
        models_dir = Path("src") / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "cal-housing-model.joblib"
        try:
            registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
           
            joblib.dump(registered_model, model_path)

        # Add a short delay to allow the model registry to stabilize before updating.
            # logging.info("Waiting for 5 seconds before adding description...")
            # time.sleep(10)
        except Exception as e:
            logging.warning(f"An error occurred during model registration or promotion: {e}")

            # Reuse the locally saved model corresponding to the best model name
            best_model_name = best_run.data.tags.get("model_name", None)
            if best_model_name:
                source_model_path = models_dir / f"{best_model_name}.joblib"
                if source_model_path.exists():
                    import shutil
                    shutil.copy(source_model_path, model_path)
                    logging.info(f"Best model copied to '{model_path}' for DVC tracking.")
                else:
                    logging.warning(f"Expected model file '{source_model_path}' not found.")
            else:
                logging.warning("Best run does not contain a 'model_name' tag.")

    
        logging.info(f"Best model saved to '{model_path}' for DVC tracking.")


        # Optionally, add a description to the registered model version
        # client.update_model_version(
        #     name=model_name,
        #     version=registered_model.version,
        #     description=f"Model from run {best_run_id}, selected as best based on R2 score."
        # )
        logging.info(f"Successfully registered model '{model_name}' version {registered_model.version}")

        


    except Exception as e:
        logging.error(f"An error occurred during model promotion: {e}")
        raise

def main():
    """Main function to run the training pipeline."""
    # Load data
    project_root = Path(__file__).resolve().parent.parent 
    data_path = project_root/"data"/"processed"/"housing.csv"
    df = pd.read_csv(str(data_path))
    
    # --- Get data version from DVC file ---
    dvc_file_path = Path(str(data_path) + ".dvc")
    data_version = "unknown"
    if dvc_file_path.exists():
        try:
            with open(dvc_file_path, 'r') as f:
                # DVC files are YAML, so using yaml library
                dvc_info = yaml.safe_load(f)
                if dvc_info:
                    data_version = dvc_info.get('outs', [{}])[0].get('md5', 'unknown')
        except Exception as e:
            logging.warning(f"Could not read DVC file to get data version: {e}")
            data_version = "dvc_file_unreadable"
    logging.info(f"Using data version (MD5 hash): {data_version}")
    
    # Define features and target
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    
    # Split data
    RANDOM_STATE = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # Define models to train
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=10),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100, max_depth=10)
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        train_and_evaluate(model, X_train, y_train, X_test, y_test, name, data_version)
        
    logging.info("All models have been trained and evaluated.")
    # Select and register the best model
    select_best_model("California Housing Prediction", "BestHousingModel")
    logging.info("Best model has been selected and registered.")
if __name__ == "__main__":
    main()
    # main1()
