# Dockerfile

# --- Stage 1: Use an official Python runtime as a parent image ---
# Using a specific version is good practice for reproducibility.
# The 'slim' variant is smaller and good for production.
FROM python:3.10-slim

# --- Set the working directory inside the container ---
# This is where our application code will live.
WORKDIR /app

# --- Copy dependency files ---
# Copy the requirements file first to leverage Docker's layer caching.
# If requirements.txt doesn't change, Docker won't re-install dependencies on subsequent builds.
COPY requirements.txt .

# --- Install dependencies ---
# We use --no-cache-dir to keep the image size small.
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy the application source code into the container ---
# This copies the 'src' directory from our project into the '/app/src' directory inside the container.
COPY ./src ./src

# --- Expose the port the app runs on ---
# This tells Docker that the container will listen on port 8000.
EXPOSE 8000

# --- Define the command to run the application ---
# This is the command that will be executed when the container starts.
# It starts the Uvicorn server, making it accessible from outside.
# Using "0.0.0.0" as the host is crucial to make it accessible outside the container.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
