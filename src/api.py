# src/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictFloat
import mlflow
import pandas as pd
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the MLflow Tracking URI
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("http://host.docker.internal:5000")


# --- 1. Initialize FastAPI app ---
app = FastAPI(
    title="California Housing Price Prediction API",
    description="An API to predict housing prices in California using a registered regression ML model.",
    version="1.0.0"
)

# --- 2. Load the MLflow Model ---
MODEL_NAME = "BestHousingModel"
MODEL_STAGE = "None" # We'll use the latest version, but you could use "Staging" or "Production"

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

# --- 4. Create the Prediction Endpoint ---
@app.post("/predict")
def predict(data: HousingData):
    """
    Accepts housing data in JSON format and returns a prediction.
    
    - **data**: A JSON object matching the HousingData schema.
    
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
        
        logging.info(f"Prediction successful. Input: {data.dict()}, Prediction: {predicted_value}")
        
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

