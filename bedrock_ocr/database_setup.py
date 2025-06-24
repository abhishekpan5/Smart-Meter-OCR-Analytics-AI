import sqlite3
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_database_tables(database_path: str = "smart_meter_database.db"):
    """Create database tables for Bedrock OCR system"""
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Create meter reference data table (if not exists)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meter_reference_data (
                meter_serial_number TEXT PRIMARY KEY,
                meter_type TEXT NOT NULL,
                current_reading TEXT NOT NULL,
                installation_date TEXT,
                last_reading_date TEXT
            )
        """)
        
        # Create processing results table (if not exists)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                extracted_serial TEXT,
                extracted_reading TEXT,
                extracted_type TEXT,
                extracted_date TEXT,
                manual_serial TEXT,
                manual_reading TEXT,
                manual_type TEXT,
                manual_date TEXT,
                serial_match_type TEXT,
                serial_confidence REAL,
                reading_validated BOOLEAN,
                reading_confidence REAL,
                processing_outcome TEXT,
                ai_confidence REAL,
                extraction_notes TEXT,
                processing_timestamp TEXT
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processing_timestamp 
            ON processing_results(processing_timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processing_outcome 
            ON processing_results(processing_outcome)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_extracted_serial 
            ON processing_results(extracted_serial)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database tables created successfully in {database_path}")
        
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def load_sample_data(database_path: str = "smart_meter_database.db"):
    """Load sample meter reference data"""
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Sample meter data
        sample_data = [
            ("12345678", "Electric", "12345.67", "2023-01-15", "2024-01-15"),
            ("87654321", "Gas", "9876.54", "2023-02-20", "2024-01-10"),
            ("11223344", "Water", "5432.10", "2023-03-10", "2024-01-05"),
            ("55667788", "Electric", "8765.43", "2023-04-05", "2024-01-12"),
            ("99887766", "Gas", "2345.67", "2023-05-12", "2024-01-08"),
            ("00112233", "Electric", "6543.21", "2023-06-18", "2024-01-14"),
            ("44556677", "Water", "3456.78", "2023-07-22", "2024-01-11"),
            ("88990011", "Gas", "7890.12", "2023-08-30", "2024-01-09"),
            ("22334455", "Electric", "4567.89", "2023-09-14", "2024-01-13"),
            ("66778899", "Water", "5678.90", "2023-10-25", "2024-01-07")
        ]
        
        # Insert sample data
        cursor.executemany("""
            INSERT OR REPLACE INTO meter_reference_data 
            (meter_serial_number, meter_type, current_reading, installation_date, last_reading_date)
            VALUES (?, ?, ?, ?, ?)
        """, sample_data)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Sample data loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        raise

def check_database_status(database_path: str = "smart_meter_database.db"):
    """Check database status and table counts"""
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Check meter reference data count
        cursor.execute("SELECT COUNT(*) FROM meter_reference_data")
        meter_count = cursor.fetchone()[0]
        
        # Check processing results count
        cursor.execute("SELECT COUNT(*) FROM processing_results")
        processing_count = cursor.fetchone()[0]
        
        # Check table structure
        cursor.execute("PRAGMA table_info(meter_reference_data)")
        meter_columns = cursor.fetchall()
        
        cursor.execute("PRAGMA table_info(processing_results)")
        processing_columns = cursor.fetchall()
        
        conn.close()
        
        logger.info(f"Database Status:")
        logger.info(f"  - Meter reference records: {meter_count}")
        logger.info(f"  - Processing result records: {processing_count}")
        logger.info(f"  - Meter table columns: {len(meter_columns)}")
        logger.info(f"  - Processing table columns: {len(processing_columns)}")
        
        return {
            "meter_count": meter_count,
            "processing_count": processing_count,
            "meter_columns": len(meter_columns),
            "processing_columns": len(processing_columns)
        }
        
    except Exception as e:
        logger.error(f"Error checking database status: {e}")
        raise

if __name__ == "__main__":
    # Create database tables
    create_database_tables()
    
    # Load sample data
    load_sample_data()
    
    # Check status
    check_database_status() 