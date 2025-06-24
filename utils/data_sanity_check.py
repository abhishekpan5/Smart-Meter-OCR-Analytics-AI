import sqlite3
import pandas as pd
from datetime import datetime
import re

def perform_sanity_check(db_name='smart_meter_database.db'):
    """
    Perform comprehensive sanity checks on the meter database.
    
    This function checks:
    1. Data completeness and integrity
    2. Data type validation
    3. Range and format validation
    4. Statistical analysis
    5. Potential data quality issues
    
    Args:
        db_name (str): Name of the SQLite database to check
    """
    
    print("=" * 60)
    print("SMART METER DATA SANITY CHECK")
    print("=" * 60)
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # 1. Basic Database Information
        print("\n1. DATABASE OVERVIEW")
        print("-" * 30)
        
        # Get total number of records
        cursor.execute("SELECT COUNT(*) FROM meter_readings")
        total_records = cursor.fetchone()[0]
        print(f"Total records: {total_records}")
        
        if total_records == 0:
            print("❌ WARNING: No records found in database!")
            return
        
        # Get date range
        cursor.execute("SELECT MIN(reading_date), MAX(reading_date) FROM meter_readings")
        date_range = cursor.fetchone()
        print(f"Date range: {date_range[0]} to {date_range[1]}")
        
        # 2. Data Completeness Check
        print("\n2. DATA COMPLETENESS")
        print("-" * 30)
        
        # Check for NULL values in each column
        columns = ['meter_serial_number', 'meter_reading', 'original_reading', 'reading_unit', 'reading_date']
        for col in columns:
            cursor.execute(f"SELECT COUNT(*) FROM meter_readings WHERE {col} IS NULL")
            null_count = cursor.fetchone()[0]
            if null_count == 0:
                print(f"✅ {col}: No NULL values")
            else:
                print(f"❌ {col}: {null_count} NULL values found")
        
        # 3. Data Type and Format Validation
        print("\n3. DATA TYPE VALIDATION")
        print("-" * 30)
        
        # Check meter serial numbers
        cursor.execute("SELECT meter_serial_number FROM meter_readings")
        serial_numbers = [row[0] for row in cursor.fetchall()]
        
        # Check if serial numbers are positive integers
        invalid_serials = [s for s in serial_numbers if not isinstance(s, int) or s <= 0]
        if not invalid_serials:
            print("✅ Meter serial numbers: All valid positive integers")
        else:
            print(f"❌ Meter serial numbers: {len(invalid_serials)} invalid values")
        
        # Check meter readings
        cursor.execute("SELECT meter_reading FROM meter_readings")
        readings = [row[0] for row in cursor.fetchall()]
        
        # Check if readings are numeric and positive
        invalid_readings = [r for r in readings if not isinstance(r, (int, float)) or r < 0]
        if not invalid_readings:
            print("✅ Meter readings: All valid numeric values")
        else:
            print(f"❌ Meter readings: {len(invalid_readings)} invalid values")
        
        # Check reading units
        cursor.execute("SELECT DISTINCT reading_unit FROM meter_readings")
        units = [row[0] for row in cursor.fetchall()]
        print(f"✅ Reading units found: {units}")
        
        # 4. Date Format Validation
        print("\n4. DATE FORMAT VALIDATION")
        print("-" * 30)
        
        cursor.execute("SELECT reading_date FROM meter_readings")
        dates = [row[0] for row in cursor.fetchall()]
        
        valid_dates = 0
        invalid_dates = []
        
        for date_str in dates:
            try:
                # Try to parse the date string
                datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                valid_dates += 1
            except ValueError:
                invalid_dates.append(date_str)
        
        print(f"✅ Valid date formats: {valid_dates}/{len(dates)}")
        if invalid_dates:
            print(f"❌ Invalid date formats: {invalid_dates}")
        
        # 5. Statistical Analysis
        print("\n5. STATISTICAL ANALYSIS")
        print("-" * 30)
        
        if readings:
            print(f"Reading statistics:")
            print(f"  - Minimum: {min(readings)}")
            print(f"  - Maximum: {max(readings)}")
            print(f"  - Average: {sum(readings)/len(readings):.2f}")
            print(f"  - Total records: {len(readings)}")
        
        # Check for duplicate readings
        cursor.execute("SELECT meter_reading, COUNT(*) FROM meter_readings GROUP BY meter_reading HAVING COUNT(*) > 1")
        duplicates = cursor.fetchall()
        if duplicates:
            print(f"⚠️  Duplicate readings found: {len(duplicates)} values appear multiple times")
            for reading, count in duplicates:
                print(f"    - Reading {reading}: {count} times")
        else:
            print("✅ No duplicate readings found")
        
        # 6. Data Consistency Checks
        print("\n6. DATA CONSISTENCY")
        print("-" * 30)
        
        # Check if original_reading matches meter_reading (when converted to numeric)
        cursor.execute("SELECT meter_reading, original_reading FROM meter_readings")
        consistency_issues = 0
        
        for reading, original in cursor.fetchall():
            try:
                original_numeric = float(original)
                if abs(reading - original_numeric) > 0.001:  # Allow for small floating point differences
                    consistency_issues += 1
            except ValueError:
                consistency_issues += 1
        
        if consistency_issues == 0:
            print("✅ Original readings match numeric readings")
        else:
            print(f"❌ {consistency_issues} consistency issues found between original and numeric readings")
        
        # 7. Business Logic Validation
        print("\n7. BUSINESS LOGIC VALIDATION")
        print("-" * 30)
        
        # Check for reasonable meter reading ranges (typical residential meters)
        reasonable_readings = [r for r in readings if 0 <= r <= 999999]
        if len(reasonable_readings) == len(readings):
            print("✅ All readings within reasonable range (0-999,999)")
        else:
            print(f"⚠️  {len(readings) - len(reasonable_readings)} readings outside typical range")
        
        # Check for sequential meter serial numbers
        if len(serial_numbers) > 1:
            serial_diff = max(serial_numbers) - min(serial_numbers)
            if serial_diff == len(serial_numbers) - 1:
                print("✅ Meter serial numbers are sequential")
            else:
                print(f"⚠️  Meter serial numbers are not sequential (gap: {serial_diff})")
        
        # 8. Summary and Recommendations
        print("\n8. SUMMARY")
        print("-" * 30)
        
        issues_found = 0
        if invalid_serials:
            issues_found += 1
        if invalid_readings:
            issues_found += 1
        if invalid_dates:
            issues_found += 1
        if duplicates:
            issues_found += 1
        if consistency_issues > 0:
            issues_found += 1
        
        if issues_found == 0:
            print("🎉 SANITY CHECK PASSED: Data appears to be clean and consistent!")
        else:
            print(f"⚠️  SANITY CHECK: {issues_found} potential issues identified")
            print("\nRecommendations:")
            print("- Review and fix any NULL values")
            print("- Validate date formats")
            print("- Check for data entry errors")
            print("- Verify business rules compliance")
        
        # Close connection
        conn.close()
        
    except Exception as e:
        print(f"❌ Error during sanity check: {str(e)}")

def main():
    """Main function to run the sanity check."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Perform sanity checks on meter database')
    parser.add_argument('--db_name', default='smart_meter_database.db', 
                       help='Name of the SQLite database to check')
    
    args = parser.parse_args()
    
    perform_sanity_check(args.db_name)

if __name__ == "__main__":
    main() 