import pandas as pd
import sqlite3
import os
import argparse
from datetime import datetime

def create_smart_meter_database(excel_file, db_name='smart_meter_database.db'):
    """
    Create a SQLite database for smart meter data from an Excel file.
    
    This function:
    1. Reads meter data from an Excel file
    2. Creates a SQLite database with appropriate schema
    3. Handles meter_reading as a numeric value in the database
    4. Preserves original reading format (for readings with leading zeros)
    5. Properly formats date/time values
    
    Args:
        excel_file (str): Path to the Excel file containing meter data
        db_name (str): Name of the SQLite database to create
        
    Returns:
        str: Path to the created database
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)
        print(f"Successfully read {len(df)} records from {excel_file}")
        
        # Store original reading as text to preserve any leading zeros
        df['original_reading'] = df['meter_reading'].astype(str)
        
        # Ensure meter_reading is numeric
        df['meter_reading'] = pd.to_numeric(df['meter_reading'])
        
        # Create a connection to the SQLite database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Drop the table if it exists
        cursor.execute("DROP TABLE IF EXISTS meter_readings")
        
        # Create the table with appropriate data types
        create_table_query = """
        CREATE TABLE meter_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meter_serial_number INTEGER NOT NULL,
            meter_reading NUMERIC NOT NULL,  -- Stored as numeric for calculations
            original_reading TEXT,           -- Preserved original format
            reading_unit TEXT,
            reading_date TEXT                -- SQLite doesn't have a native datetime type
        );
        """
        
        cursor.execute(create_table_query)
        print("Created meter_readings table")
        
        # Insert the data into the table
        for _, row in df.iterrows():
            insert_query = """
            INSERT INTO meter_readings 
            (meter_serial_number, meter_reading, original_reading, reading_unit, reading_date) 
            VALUES (?, ?, ?, ?, ?)
            """
            
            cursor.execute(insert_query, (
                int(row['meter_serial_number']),
                float(row['meter_reading']),
                row['original_reading'],
                row['reading unit'],
                str(row['reading_date'])
            ))
        
        # Commit the changes
        conn.commit()
        print(f"Inserted {len(df)} records into the database")
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX idx_meter_serial_number ON meter_readings(meter_serial_number)")
        cursor.execute("CREATE INDEX idx_reading_date ON meter_readings(reading_date)")
        conn.commit()
        print("Created indexes for faster queries")
        
        # Verify the data
        cursor.execute("SELECT * FROM meter_readings")
        rows = cursor.fetchall()
        
        print("\nSample data in the database:")
        for row in rows[:5]:  # Show up to 5 rows
            print(row)
        
        # Get and display the table schema
        cursor.execute("PRAGMA table_info(meter_readings)")
        schema = cursor.fetchall()
        
        print("\nTable schema:")
        for col in schema:
            print(f"Column: {col[1]}, Type: {col[2]}")
        
        # Close the connection
        conn.close()
        
        return db_name
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a SQLite database from an Excel file containing meter data')
    parser.add_argument('excel_file', help='Path to the Excel file containing meter data')
    parser.add_argument('--db_name', default='smart_meter_database.db', help='Name of the SQLite database to create')
    
    args = parser.parse_args()
    
    # Create the database
    db_path = create_smart_meter_database(args.excel_file, args.db_name)
    
    if db_path:
        print(f"\nDatabase created successfully: {db_path}")
    else:
        print("\nFailed to create database")

if __name__ == "__main__":
    main()
