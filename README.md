# Health Impact Prediction - Air Quality Analysis

## Project Overview
This project explores the relationship between environmental conditions and health risks through a comprehensive machine learning approach. It analyzes how air quality and weather parameters affect health risk levels, providing predictive models that can be used for public health planning and early warning systems.

## Dataset
The analysis uses the "Urban Air Quality and Health Impact Dataset" which contains:
- Environmental measurements (temperature, humidity, air pressure, wind speed)
- Location data (cities)
- Temporal information (Fall season)
- Health risk scores and categories

## Methodology
The project follows a structured data science workflow:

1. **Exploratory Data Analysis**
   - Statistical analysis of key features
   - Correlation studies
   - Geographic and temporal pattern analysis
   - Visualization of feature relationships

2. **Feature Engineering**
   - Feature selection based on correlation analysis
   - Creation of categorical risk levels from continuous health risk scores
   - Preprocessing of numerical and categorical variables

3. **Model Development**
   - Implementation of multiple classification algorithms:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
   - Cross-validation for robust performance estimation
   - Hyperparameter tuning to optimize model performance

4. **Evaluation**
   - Comprehensive metrics: accuracy, precision, recall, F1 score
   - ROC curve analysis
   - Confusion matrices
   - Statistical significance testing

## Key Findings

### Model Performance
- The Random Forest model performed best with ~90% accuracy after hyperparameter optimization
- Logistic Regression showed competitive performance despite its simplicity
- All models effectively distinguished between extreme risk categories (Low vs. Very High)

### Important Environmental Factors
- **Severity Score**: Strongest predictor of health risk
- **Heat Index**: Strong positive relationship with health outcomes
- **Temperature parameters**: Consistently associated with elevated health risks
- **Wind parameters**: Significant influence on health risk levels
- **Humidity and Dew Point**: Important factors in respiratory effects

### Geographic Insights
- Different cities show markedly different health risk profiles
- Phoenix had the highest average risk scores
- San Jose showed the lowest risk levels
- Local environmental conditions significantly impact health outcomes

## Limitations
- Analysis is limited to Fall season data only
- More comprehensive year-round observations would provide deeper seasonal insights

## Usage
The trained model has been saved and can be used for predictions:

```python
import joblib

# Load the saved model
model = joblib.load('best_model.joblib')

# Sample prediction (requires the same feature set used during training)
prediction = model.predict(sample_data)
```

## Applications
- **Public Health Planning**: Forecast risk levels based on weather predictions
- **Early Warning Systems**: Alert vulnerable populations about potential health risks
- **City-Specific Strategies**: Tailor approaches to each city's environmental profile
- **Resource Allocation**: Focus healthcare resources on high-risk periods and locations

## Future Directions
- Expand data collection across all seasons
- Develop city-specific predictive models
- Incorporate additional environmental metrics (air quality, pollen, pollutants)
- Build time-series forecasting for advanced warning
- Create interactive dashboards for public health officials

## Files
- `health_impact_prediction.py`: Main analysis script with all code and documentation
- `best_model.joblib`: Saved machine learning model for predictions
- `model_metadata.json`: Information about the model, features, and risk mappings

This project demonstrates the effective application of machine learning to predict health risks based on environmental data, providing valuable tools for public health management and policy decisions. 
