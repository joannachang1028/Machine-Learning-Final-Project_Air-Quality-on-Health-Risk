import pandas as pd
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os

# Database connection parameters - modify these as needed
DB_NAME = "air_quality_db"
DB_USER = "postgres"
DB_PASSWORD = "postgres"  # Change this to your actual PostgreSQL password
DB_HOST = "localhost"
DB_PORT = "5432"

# CSV file path
CSV_FILE = "Urban Air Quality and Health Impact Dataset.csv"

def create_database():
    """Create the PostgreSQL database if it doesn't exist"""
    # Connect to default 'postgres' database first
    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_NAME,))
    exists = cursor.fetchone()
    
    if not exists:
        print(f"Creating database {DB_NAME}...")
        cursor.execute(f"CREATE DATABASE {DB_NAME}")
        print(f"Database {DB_NAME} created successfully!")
    else:
        print(f"Database {DB_NAME} already exists.")
    
    cursor.close()
    conn.close()

def create_table_and_import_data():
    """Create table and import data from CSV file"""
    # Read CSV to get column names and types
    df = pd.read_csv(CSV_FILE)
    
    # Clean the data first
    # Handle missing values and convert types appropriately
    df = df.replace({pd.NA: None})
    
    # Make sure Is_Weekend is properly converted to boolean
    if 'Is_Weekend' in df.columns:
        df['Is_Weekend'] = df['Is_Weekend'].map({'TRUE': True, 'FALSE': False})
    
    # Connect to the new database
    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Drop existing table if needed
    cursor.execute("DROP TABLE IF EXISTS air_quality_data")
    
    # Create the table with appropriate data types
    create_table_query = """
    CREATE TABLE IF NOT EXISTS air_quality_data (
        id SERIAL PRIMARY KEY,
        datetime DATE,
        datetimeEpoch FLOAT,
        tempmax FLOAT,
        tempmin FLOAT,
        temp FLOAT,
        feelslikemax FLOAT,
        feelslikemin FLOAT,
        feelslike FLOAT,
        dew FLOAT,
        humidity FLOAT,
        precip FLOAT,
        precipprob FLOAT,
        precipcover FLOAT,
        preciptype TEXT,
        snow FLOAT,
        snowdepth FLOAT,
        windgust FLOAT,
        windspeed FLOAT,
        winddir FLOAT,
        pressure FLOAT,
        cloudcover FLOAT,
        visibility FLOAT,
        solarradiation FLOAT,
        solarenergy FLOAT,
        uvindex FLOAT,
        severerisk FLOAT,
        sunrise TEXT,
        sunriseEpoch FLOAT,
        sunset TEXT,
        sunsetEpoch FLOAT,
        moonphase FLOAT,
        conditions TEXT,
        description TEXT,
        icon TEXT,
        stations TEXT,
        source TEXT,
        city TEXT,
        temp_range FLOAT,
        heat_index FLOAT,
        severity_score FLOAT,
        condition_code FLOAT,
        month INTEGER,
        season TEXT,
        day_of_week TEXT,
        is_weekend BOOLEAN,
        health_risk_score FLOAT
    );
    """
    
    cursor.execute(create_table_query)
    print("Table created successfully.")
    
    # Since the CSV column names might not match PostgreSQL column naming conventions
    # Map CSV column names to Postgres column names (all lowercase)
    column_mapping = {col: col.lower() for col in df.columns}
    df = df.rename(columns=column_mapping)
    
    # Import data row by row
    print("Importing data...")
    
    # Create column list and placeholders for prepared statement
    columns = list(df.columns)
    placeholders = ', '.join(['%s'] * len(columns))
    insert_query = f"INSERT INTO air_quality_data ({', '.join(columns)}) VALUES ({placeholders})"
    
    # Process each row
    for i, row in df.iterrows():
        values = []
        for col in columns:
            val = row[col]
            # Special handling for boolean is_weekend column
            if col == 'is_weekend':
                if val == 'TRUE' or val is True or val == 1:
                    val = True
                elif val == 'FALSE' or val is False or val == 0:
                    val = False
                else:
                    val = None
            values.append(val)
        
        try:
            cursor.execute(insert_query, tuple(values))
            if (i + 1) % 100 == 0:
                print(f"Imported {i + 1} rows...")
        except Exception as e:
            print(f"Error importing row {i}: {e}")
            print(f"Row data: {row}")
            conn.rollback()
            continue
    
    print(f"Data import completed! Total rows: {len(df)}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file {CSV_FILE} not found.")
        exit(1)
    
    try:
        create_database()
        create_table_and_import_data()
        print("Database setup complete!")
    except Exception as e:
        print(f"Error: {e}") 