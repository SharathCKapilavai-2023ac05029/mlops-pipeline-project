# src/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictFloat
import mlflow
import pandas as pd
import logging

# Set up basic logging
import sqlite3
import json
from datetime import datetime

# Prometheus client for monitoring
from starlette.responses import Response
from prometheus_client import Counter, Histogram, generate_latest
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the MLflow Tracking URI
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("http://host.docker.internal:5000")

# --- Database Setup ---
DB_PATH = "logs/predictions.db"

def init_db():
    """Initializes the SQLite database and creates the predictions table if it doesn't exist."""
    try:
        # The logs directory will be mapped via a Docker volume
        Path("logs").mkdir(exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                input_data TEXT NOT NULL,
                prediction REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logging.info(f"Database initialized successfully at {DB_PATH}")
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
        raise

# --- 1. Initialize FastAPI app ---
app = FastAPI(
    title="California Housing Price Prediction API",
    description="An API to predict housing prices in California using a registered regression ML model.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Run database initialization on API startup."""
    init_db()

# --- 2. Load the MLflow Model ---
MODEL_NAME = "BestHousingModel"
MODEL_STAGE = "None"

try:
    logging.info(f"Loading model '{MODEL_NAME}' version '{MODEL_STAGE}' from MLflow Model Registry...")
    # The model URI format loads the latest version of the model in the specified stage
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    # If the model fails to load, we can't start the API.
    # In a real-world scenario, you might have fallback logic or alerts.
    raise RuntimeError(f"Could not load model '{MODEL_NAME}'. Shutting down.") from e


# --- 3. Define the Input Schema using Pydantic ---
# This creates a data contract for our API input.

class HousingData(BaseModel):
    MedInc: StrictFloat = Field(..., example=3.8716, description="Median income in block group")
    HouseAge: StrictFloat = Field(..., example=23.0, ge=0, description="Median house age in block group (must be non-negative)")
    AveRooms: StrictFloat = Field(..., example=5.82, ge=0, description="Average number of rooms (must be non-negative)")
    AveBedrms: StrictFloat = Field(..., example=1.09, ge=0, description="Average number of bedrooms (must be non-negative)")
    Population: StrictFloat = Field(..., example=1234.0, ge=0, description="Block group population (must be non-negative)")
    AveOccup: StrictFloat = Field(..., example=3.01, ge=0, description="Average household occupancy (must be non-negative)")
    Latitude: StrictFloat = Field(..., example=34.23, description="Block group latitude")
    Longitude: StrictFloat = Field(..., example=-118.45, description="Block group longitude")

# ---Prometheus Metrics ---
PREDICTION_COUNTER = Counter("predictions_total", "Total number of predictions made")
PREDICTION_VALUE_HISTOGRAM = Histogram("prediction_value_usd", "Histogram of predicted housing values (in $100,000s)")

# --- 4. Create the Prediction Endpoint ---
@app.post("/predict")
def predict(data: HousingData):
    """
    Accepts housing data in JSON format and returns a prediction.
        - **data**: A JSON object matching the HousingData schema.
    Logs the request and response to a SQLite database.
    Returns:
        - A JSON object with the prediction.
    """
    try:
        # Convert the Pydantic model to a pandas DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Make a prediction
        prediction = model.predict(input_df)
        
        # The output of a scikit-learn model is often a numpy array, so we extract the first element.
        predicted_value = prediction[0]
        
        # --- Logging and Monitoring ---
        # Log the request and response to the SQLite database
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO predictions (timestamp, input_data, prediction) VALUES (?, ?, ?)",
                (datetime.now(), json.dumps(data.dict()), predicted_value)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            # Log to console if DB logging fails, but don't crash the request
            logging.error(f"Failed to log prediction to database: {e}")

        logging.info(f"Prediction successful. Input: {data.dict()}, Prediction: {predicted_value}")

         # Update Prometheus metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_VALUE_HISTOGRAM.observe(predicted_value)
        
        return {"prediction": predicted_value}

    except Exception as e:
        logging.error(f"Prediction failed. Error: {e}")
        # This will return a 500 Internal Server Error to the client
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

# --- 5. Add a Root Endpoint for Health Checks ---
@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"status": "API is running"}

# --- 6. Add a Metrics Endpoint for Prometheus ---
@app.get("/metrics")
def get_metrics():
    """Exposes Prometheus metrics."""
    return Response(media_type="text/plain", content=generate_latest())