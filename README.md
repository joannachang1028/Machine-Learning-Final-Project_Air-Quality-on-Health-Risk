# Air Quality Health Risk Prediction System

This project creates a full-stack application for predicting health risks based on air quality and weather data. It includes:

1. A PostgreSQL database for storing air quality data
2. Machine learning models for health risk prediction
3. A Streamlit web interface for data exploration and predictions

## Prerequisites

- Python 3.7+
- PostgreSQL 16
- Required Python packages: pandas, psycopg2, streamlit, scikit-learn, matplotlib, seaborn, joblib

## Setup Instructions

### 1. Install Required Packages

```bash
pip install pandas psycopg2-binary streamlit scikit-learn matplotlib seaborn joblib
```

### 2. PostgreSQL Setup

Ensure PostgreSQL 16 is installed and running on your system. You may need to create a user and password for the database.

### 3. Database Configuration

Edit the database connection parameters in `create_database.py` and `app.py`:

```python
DB_NAME = "air_quality_db"
DB_USER = "postgres"  # Change to your PostgreSQL username
DB_PASSWORD = "postgres"  # Change to your PostgreSQL password
DB_HOST = "localhost"
DB_PORT = "5432"
```

### 4. Import Data into PostgreSQL

Run the database setup script to create a database and import data from the CSV file:

```bash
python create_database.py
```

### 5. Train and Save the ML Model

Execute the notebook or Python script `health_impact_prediction.py` to train the machine learning model. Make sure to call the `save_best_model()` function after model training:

```python
# After training your best model, save it for the web application
best_model = models_results['Best Model Name']['model']
save_best_model(best_model, risk_mapping, X_train.columns)
```

This will save:
- `best_model.joblib`: The trained model
- `model_metadata.json`: Model metadata including risk mappings and feature names

### 6. Run the Web Application

Start the Streamlit web application:

```bash
streamlit run UI.py
```

The application will be available at http://localhost:8501

## Application Features

1. **Home Page**: Overview of the application and recent data
2. **Data Explorer**: Filter and visualize air quality data
3. **Make Prediction**: Enter parameters to predict health risk levels
4. **About**: Information about the application

## Database Structure

The PostgreSQL database contains a table called `air_quality_data` with the following key columns:

- `datetime`: Date of the air quality measurement
- `temp`, `humidity`, `pressure`, etc.: Weather measurements
- `health_risk_score`: Target variable for prediction

## Model Information

The machine learning model predicts health risk categories based on air quality and weather parameters. The risk categories range from Low to Extreme.

## Troubleshooting

- **Database Connection Issues**: Ensure PostgreSQL is running and the connection parameters are correct
- **Model Loading Errors**: Verify that the model has been trained and saved correctly
- **Missing Data**: Confirm that the CSV data was imported successfully