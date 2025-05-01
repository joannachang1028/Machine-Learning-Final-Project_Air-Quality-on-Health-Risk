import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import joblib
from psycopg2.extras import RealDictCursor
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Database connection parameters
DB_NAME = "air_quality_db"
DB_USER = "postgres"
DB_PASSWORD = "postgres"  # Change this to your actual PostgreSQL password
DB_HOST = "localhost"
DB_PORT = "5432"

# Load the trained model
@st.cache_resource  # Cache the model to avoid reloading on every run
def load_model():
    try:
        model = joblib.load('best_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is trained and saved correctly.")
        return None

# Load model metadata
@st.cache_data
def load_metadata():
    try:
        with open('model_metadata.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Model metadata file not found.")
        return {"risk_mapping": {
            'Low': 0,
            'Moderate': 1,
            'High': 2,
            'Very High': 3
        }}

# Database connection function
def get_db_connection():
    try:
        conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME
        )
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

# Function to fetch data from the database
def fetch_data(query, params=None):
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# Function to execute database operations (insert, update, delete)
def execute_db_operation(query, params=None):
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        cur = conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Database operation error: {e}")
        return False
    finally:
        conn.close()

# Function to make predictions
def predict_health_risk(input_data):
    model = load_model()
    metadata = load_metadata()
    
    if model is None:
        st.error("Model could not be loaded. Please check if the model file exists.")
        return None
    
    try:
        # Check if all expected features are present
        expected_features = metadata.get("features", [])
        if expected_features:
            missing_features = [f for f in expected_features if f not in input_data.columns]
            if missing_features:
                st.warning(f"Missing features: {', '.join(missing_features)}")
                # Add missing features with default values (0)
                for feature in missing_features:
                    input_data[feature] = 0
        
        # Ensure all features are in the correct data type
        # Convert string columns to categorical codes if needed
        for col in input_data.columns:
            if input_data[col].dtype == 'object':
                # For object/string columns, we need to handle them specially
                # If the feature is categorical with few values, we can use simple mapping
                if col == 'City':
                    city_mapping = {
                        'Phoenix': 0, 'Los Angeles': 1, 'San Jose': 2, 
                        'San Francisco': 3, 'Other': 4
                    }
                    input_data[col] = input_data[col].map(lambda x: city_mapping.get(x, 4))
                elif col == 'Season':
                    season_mapping = {
                        'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3
                    }
                    input_data[col] = input_data[col].map(lambda x: season_mapping.get(x, 2))  # Default to Fall
                elif col == 'preciptype':
                    # Handle precipitation type
                    precip_mapping = {'none': 0, 'rain': 1, 'snow': 2}
                    input_data[col] = input_data[col].map(lambda x: precip_mapping.get(x, 0))
                else:
                    # For other string columns, just convert to categorical codes
                    input_data[col] = pd.Categorical(input_data[col]).codes
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Map numeric predictions back to category labels
        risk_mapping_reverse = {v: k for k, v in metadata["risk_mapping"].items()}
        if prediction[0] in risk_mapping_reverse:
            prediction_label = risk_mapping_reverse[prediction[0]]
            return prediction_label
        else:
            st.warning(f"Prediction value {prediction[0]} not found in risk mapping.")
            # As a fallback, try to map based on closest value
            closest_key = min(metadata["risk_mapping"].values(), key=lambda x: abs(x - prediction[0]))
            return risk_mapping_reverse.get(closest_key, "Unknown")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None

# Function to calculate correlation with Health Risk Score
def calculate_correlation(feature, city=None):
    # Use lowercase for health_risk_score in the SQL query
    query = """
        SELECT {} as feature, health_risk_score 
        FROM air_quality_data
        WHERE {} IS NOT NULL
    """.format(feature, feature)
    
    if city and city != "All":
        query += " AND City = %s"
        df = fetch_data(query, (city,))
    else:
        df = fetch_data(query)
    
    if df.empty:
        return None, None
    
    # Rename column to ensure consistent casing
    if 'health_risk_score' in df.columns and 'Health_Risk_Score' not in df.columns:
        df = df.rename(columns={'health_risk_score': 'Health_Risk_Score'})
    
    try:
        correlation = df.corr().iloc[0, 1]
        return correlation, df
    except Exception as e:
        st.error(f"Error calculating correlation: {e}")
        return None, None

# Descriptive Information page
def descriptive_info_page():
    st.header("Descriptive Information - Correlation Analysis")
    
    # Get all cities for filter
    try:
        cities = ["All"] + fetch_data("SELECT DISTINCT City FROM air_quality_data ORDER BY City")["city"].tolist()
    except Exception as e:
        st.error(f"Error fetching cities: {e}")
        cities = ["All"]
    
    # Features for correlation analysis
    features = [
        "Heat_Index", "Severity_Score", "windgust", "dew", "severerisk", 
        "feelslike", "feelslikemax", "windspeed", "feelslikemin"
    ]
    
    # Filters
    st.sidebar.subheader("Filters")
    selected_city = st.sidebar.selectbox("Select City", cities)
    selected_feature = st.sidebar.selectbox("Select Feature", features)
    
    # Calculate correlation
    correlation, data = calculate_correlation(selected_feature, selected_city)
    
    if correlation is not None and data is not None and not data.empty:
        try:
            # Display correlation value
            city_text = f"in {selected_city}" if selected_city != "All" else "across all cities"
            st.subheader(f"Correlation between {selected_feature} and Health Risk Score {city_text}")
            
            # Format correlation with color based on strength
            corr_color = "green" if correlation >= 0 else "red"
            st.markdown(f"**Correlation Coefficient:** <span style='color:{corr_color};font-size:24px'>{correlation:.4f}</span>", unsafe_allow_html=True)
            
            # Correlation strength interpretation
            if abs(correlation) < 0.3:
                strength = "weak"
            elif abs(correlation) < 0.7:
                strength = "moderate"
            else:
                strength = "strong"
            
            direction = "positive" if correlation >= 0 else "negative"
            st.write(f"This indicates a {strength} {direction} correlation.")
            
            # Plot scatter plot with regression line
            if 'feature' in data.columns and 'Health_Risk_Score' in data.columns:
                st.subheader("Correlation Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.regplot(x='feature', y='Health_Risk_Score', data=data, scatter_kws={'alpha':0.5}, ax=ax)
                plt.xlabel(selected_feature)
                plt.ylabel('Health Risk Score')
                plt.title(f'{selected_feature} vs Health Risk Score')
                st.pyplot(fig)
                
                # Show the raw data
                if st.checkbox("Show Raw Data"):
                    st.dataframe(data)
            else:
                st.error(f"Required columns not found in data. Available columns: {', '.join(data.columns)}")
        except Exception as e:
            st.error(f"Error displaying correlation data: {e}")
    else:
        st.warning(f"No data available for {selected_feature} in {selected_city if selected_city != 'All' else 'any city'}.")

# Predictive Analysis page
def predictive_analysis_page():
    st.header("Predictive Analysis - Health Risk Category")
    
    # Check if model exists
    model = load_model()
    if model is None:
        st.error("Model not available. Please ensure the model is trained and saved correctly.")
        st.info("You can still input data, but predictions will not be made.")
    
    # Load model metadata to get expected features
    metadata = load_metadata()
    expected_features = metadata.get("features", [])
    
    if not expected_features:
        st.error("Model metadata does not contain feature information. Cannot make predictions.")
        return
    
    st.write("""
    This tool predicts health risk categories based on the top 10 most important features.
    Adjust the values below to see how different environmental conditions affect health risk.
    """)
    
    # Create columns for input
    col1, col2 = st.columns(2)
    
    # Numerical features (7) in first column
    with col1:
        st.subheader("Numerical Features")
        windgust = st.slider("Wind Gust", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
        severity_score = st.slider("Severity Score", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        dew = st.slider("Dew Point", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        heat_index = st.slider("Heat Index", min_value=0.0, max_value=120.0, value=75.0, step=0.1)
        feelslike = st.slider("Feels Like", min_value=0.0, max_value=120.0, value=75.0, step=0.1)
        windspeed = st.slider("Wind Speed", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
        feelslikemax = st.slider("Feels Like Max", min_value=0.0, max_value=120.0, value=80.0, step=0.1)
    
    # Categorical features (3) in second column
    with col2:
        st.subheader("Categorical Features")
        
        # Temperature category
        temperature = st.selectbox(
            "Temperature Category",
            ["Low", "Moderate", "High", "Very High"]
        )
        
        # Wind category
        wind = st.selectbox(
            "Wind Category",
            ["Calm", "Light", "Moderate", "Strong"]
        )
        
        # Moisture category
        moisture = st.selectbox(
            "Moisture Category",
            ["Dry", "Moderate", "Humid", "Very Humid"]
        )
        
        # Convert categorical to numerical
        def temp_to_num(cat):
            mapping = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3}
            return mapping[cat]
        
        def wind_to_num(cat):
            mapping = {"Calm": 0, "Light": 1, "Moderate": 2, "Strong": 3}
            return mapping[cat]
        
        def moisture_to_num(cat):
            mapping = {"Dry": 0, "Moderate": 1, "Humid": 2, "Very Humid": 3}
            return mapping[cat]
    
    # Predict button
    if st.button("Predict Health Risk"):
        if model is None:
            st.error("Cannot make prediction: Model not available")
            return
            
        try:
            # Prepare input data with only the 10 specified features
            input_data = pd.DataFrame({
                'windgust': [windgust],
                'Severity_Score': [severity_score],
                'dew': [dew],
                'Heat_Index': [heat_index],
                'feelslike': [feelslike],
                'windspeed': [windspeed],
                'feelslikemax': [feelslikemax],
                'Temperature': [temp_to_num(temperature)],
                'Wind': [wind_to_num(wind)],
                'Moisture': [moisture_to_num(moisture)]
            })
            
            # Check if we need to add other features expected by the model
            missing_features = [f for f in expected_features if f not in input_data.columns]
            
            # If other features are required by the model, add them with default values
            if missing_features:
                # Set reasonable defaults for missing features
                default_values = {
                    "temp": feelslike,
                    "tempmax": feelslikemax,
                    "tempmin": feelslike - 10.0,
                    "feelslikemin": feelslike - 10.0,
                    "humidity": 50.0 if moisture == "Moderate" else (70.0 if moisture == "Humid" else (90.0 if moisture == "Very Humid" else 30.0)),
                    "precip": 0.0,
                    "precipprob": 0.0,
                    "precipcover": 0.0,
                    "preciptype": "none",
                    "snow": 0,
                    "snowdepth": 0.0,
                    "winddir": 180,
                    "pressure": 1015.0,
                    "cloudcover": 50.0,
                    "visibility": 10.0,
                    "solarradiation": 250.0,
                    "solarenergy": 15.0,
                    "uvindex": 5,
                    "severerisk": severity_score,
                    "moonphase": 0.5,
                    "conditions": "Clear",
                    "description": "Clear conditions",
                    "icon": "clear-day",
                    "stations": "airport",
                    "source": "obs",
                    "City": "Phoenix",
                    "Temp_Range": 20.0,
                    "Condition_Code": 1,
                    "Month": 9,
                    "Season": "Fall",
                    "Day_of_Week": 3,
                    "Is_Weekend": 0
                }
                
                # Add missing features with default values
                for feature in missing_features:
                    input_data[feature] = default_values.get(feature, 0)
            
            # Display the input data for debugging
            if st.checkbox("Show input data"):
                st.write("Input data for prediction:")
                st.dataframe(input_data)
            
            # Get prediction
            prediction = predict_health_risk(input_data)
            
            # Display prediction
            if prediction:
                # Create prediction visualization
                risk_levels = ["Low", "Moderate", "High", "Very High"]
                risk_colors = ["green", "yellow", "orange", "red"]
                
                # Find the index of the predicted risk
                if prediction in risk_levels:
                    risk_index = risk_levels.index(prediction)
                    
                    # Display the prediction
                    st.markdown(f"### Predicted Health Risk: <span style='color:{risk_colors[risk_index]};font-size:28px'>{prediction}</span>", unsafe_allow_html=True)
                    
                    # Display risk guidelines
                    if prediction == "Low":
                        st.success("Low Risk: Air quality is satisfactory. Enjoy outdoor activities.")
                    elif prediction == "Moderate":
                        st.info("Moderate Risk: Acceptable air quality. Consider reducing prolonged outdoor activity if you experience unusual symptoms.")
                    elif prediction == "High":
                        st.warning("High Risk: Sensitive individuals should limit outdoor exertion.")
                    elif prediction == "Very High":
                        st.error("Very High Risk: Everyone should reduce outdoor activity. Use air purifiers indoors.")
                else:
                    st.warning(f"Received an unknown risk category: {prediction}")
            else:
                st.error("Unable to make prediction. Please check your inputs and the model.")
        except Exception as e:
            st.error(f"Error in prediction process: {e}")
            st.info("Please check if your input features match the model's expectations.")

# Database Management page
def database_management_page():
    st.header("Database Management - Add/Delete Data")
    
    # Check database connection first
    conn = get_db_connection()
    if conn is None:
        st.error("Cannot connect to database. Please check your database configuration.")
        return
    conn.close()
    
    # Create tabs for Add and Delete operations
    tab1, tab2 = st.tabs(["Add Data", "Delete Data"])
    
    # Add Data tab
    with tab1:
        st.subheader("Add New Air Quality Data")
        
        with st.form("add_data_form"):
            # Columns for better layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                city = st.text_input("City")
                temp = st.number_input("Temperature (°F)", value=75.0)
                humidity = st.number_input("Humidity (%)", value=50.0)
                windspeed = st.number_input("Wind Speed", value=10.0)
                heat_index = st.number_input("Heat Index", value=80.0)
            
            with col2:
                windgust = st.number_input("Wind Gust", value=15.0)
                dew = st.number_input("Dew Point", value=50.0)
                feelslike = st.number_input("Feels Like", value=75.0)
                feelslikemax = st.number_input("Feels Like Max", value=80.0)
                feelslikemin = st.number_input("Feels Like Min", value=70.0)
            
            with col3:
                severity_score = st.number_input("Severity Score", value=3.0)
                severerisk = st.number_input("Severe Risk", value=0.0)
                season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
                health_risk_score = st.number_input("Health Risk Score", value=50.0)
            
            submitted = st.form_submit_button("Add Data")
            
            if submitted:
                try:
                    # Get actual column names from database to handle case sensitivity
                    table_info = fetch_data("SELECT column_name FROM information_schema.columns WHERE table_name = 'air_quality_data'")
                    if table_info.empty:
                        raise Exception("Could not fetch table structure")
                    
                    column_names = table_info['column_name'].tolist()
                    
                    # Build query based on actual column names
                    columns = []
                    placeholders = []
                    values = []
                    
                    # Map our input field names to database column names (case-insensitive)
                    field_mapping = {
                        'city': city, 'temp': temp, 'humidity': humidity, 
                        'windspeed': windspeed, 'heat_index': heat_index, 
                        'windgust': windgust, 'dew': dew, 
                        'feelslike': feelslike, 'feelslikemax': feelslikemax, 
                        'feelslikemin': feelslikemin, 'severity_score': severity_score,
                        'severerisk': severerisk, 'season': season, 
                        'health_risk_score': health_risk_score
                    }
                    
                    # Match our fields with actual column names (case-insensitive)
                    for db_col in column_names:
                        for field, value in field_mapping.items():
                            if db_col.lower() == field.lower():
                                columns.append(db_col)
                                placeholders.append('%s')
                                values.append(value)
                                break
                    
                    # Create the query
                    insert_query = f"""
                    INSERT INTO air_quality_data (
                        {", ".join(columns)}
                    ) VALUES (
                        {", ".join(placeholders)}
                    )
                    """
                    
                    success = execute_db_operation(insert_query, tuple(values))
                    
                    if success:
                        st.success("Data added successfully!")
                    else:
                        st.error("Failed to add data. Please check the inputs and try again.")
                        
                except Exception as e:
                    st.error(f"Error preparing the database operation: {e}")
    
    # Delete Data tab
    with tab2:
        st.subheader("Delete Air Quality Data")
        
        # Provide options to filter what to delete
        delete_option = st.radio(
            "Delete by:",
            ["City", "ID"]
        )
        
        if delete_option == "City":
            try:
                # Get all cities for selection
                cities = fetch_data("SELECT DISTINCT City FROM air_quality_data ORDER BY City")["city"].tolist()
                if cities:
                    selected_city = st.selectbox("Select City to Delete", cities)
                    
                    if st.button("Show Data for Selected City"):
                        # Show data that will be deleted
                        city_data = fetch_data("SELECT * FROM air_quality_data WHERE City = %s LIMIT 100", (selected_city,))
                        if not city_data.empty:
                            st.dataframe(city_data)
                            st.warning(f"Found {len(city_data)} records for {selected_city}. The delete operation may affect more records than shown.")
                        else:
                            st.info(f"No data found for {selected_city}.")
                    
                    # Add confirmation for deletion
                    confirm = st.checkbox("I confirm I want to delete this data")
                    
                    if st.button("Delete Data", disabled=not confirm):
                        # Delete data for selected city
                        delete_query = "DELETE FROM air_quality_data WHERE City = %s"
                        success = execute_db_operation(delete_query, (selected_city,))
                        
                        if success:
                            st.success(f"Data for {selected_city} deleted successfully!")
                        else:
                            st.error("Failed to delete data. Please try again.")
                else:
                    st.info("No city data found in the database.")
            except Exception as e:
                st.error(f"Error loading cities: {e}")
        
        elif delete_option == "ID":
            record_id = st.number_input("Enter Record ID to Delete", min_value=1, step=1)
            
            if st.button("Show Data for Selected ID"):
                # Show data that will be deleted
                id_data = fetch_data("SELECT * FROM air_quality_data WHERE id = %s", (record_id,))
                if not id_data.empty:
                    st.dataframe(id_data)
                else:
                    st.info(f"No data found for ID {record_id}.")
            
            # Add confirmation for deletion
            confirm = st.checkbox("I confirm I want to delete this record")
            
            if st.button("Delete Record", disabled=not confirm):
                # Delete record with selected ID
                delete_query = "DELETE FROM air_quality_data WHERE id = %s"
                success = execute_db_operation(delete_query, (record_id,))
                
                if success:
                    st.success(f"Record with ID {record_id} deleted successfully!")
                else:
                    st.error("Failed to delete record. Please try again.")

# Main app function
def main():
    st.title("Air Quality Health Risk Analysis")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a function",
        ["Home", "Descriptive Information", "Predictive Analysis", "Database Management"]
    )
    
    # Home page
    if app_mode == "Home":
        st.header("Welcome to the Air Quality Health Risk Analysis App")
        
        st.markdown("""
        This application provides tools to analyze and predict health risks related to air quality 
        and weather conditions. Use the sidebar to navigate between different functions:
        
        ### 1. Descriptive Information
        Explore how different environmental factors correlate with health risks.
        Filter by city and variables like Heat Index, Severity Score, and more.
        
        ### 2. Predictive Analysis
        Predict health risk categories using our trained machine learning model.
        The model uses the top 10 most important features identified in our analysis.
        
        ### 3. Database Management
        Add new data points or delete existing records from the database.
        
        ---
        
        **Getting Started:** Select a function from the sidebar to begin exploring!
        """)
        
        # Show some sample data
        try:
            st.subheader("Sample Data Preview")
            sample_data = fetch_data("SELECT * FROM air_quality_data ORDER BY RANDOM() LIMIT 5")
            if not sample_data.empty:
                st.dataframe(sample_data)
            else:
                st.info("No data available. Please add data using the Database Management function.")
        except Exception as e:
            st.error("Error loading sample data. Please check the database connection.")
    
    # Descriptive Information page
    elif app_mode == "Descriptive Information":
        descriptive_info_page()
    
    # Predictive Analysis page
    elif app_mode == "Predictive Analysis":
        predictive_analysis_page()
    
    # Database Management page
    elif app_mode == "Database Management":
        database_management_page()

# Run the app
if __name__ == "__main__":
    main() 