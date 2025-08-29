##📊 MLflow Project

This repository contains an MLflow project for managing the full machine learning lifecycle, including experimentation, reproducibility, and deployment.

🚀 Features

Experiment tracking with MLflow

Reproducible runs using MLproject

Model packaging and deployment

Parameterized training

Logging of metrics, artifacts, and models

##📂 Project Structure
.
├── MLproject              # MLflow project definition
├── conda.yaml             # Conda environment for reproducibility
├── train.py               # Training script (entry point)
├── utils.py               # Helper functions
├── data/                  # Input data (git-ignored if large)
├── models/                # Saved models
└── README.md              # Project documentation

⚙️ Requirements

Python 3.10+

MLflow

Conda or virtualenv

Install dependencies:

pip install mlflow


##📊 Tracking Experiments

Start the MLflow tracking UI:

mlflow ui


Open in browser: http://localhost:5000

##📦 Saving and Loading Models

Train and log model:

import mlflow.sklearn
mlflow.sklearn.log_model(model, "model")


Load model for inference:

model = mlflow.sklearn.load_model("runs:/<run_id>/model")

##🚀 Deployment

Export model for serving:

mlflow models serve -m runs:/<run_id>/model -p 1234


