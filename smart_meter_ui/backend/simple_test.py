#!/usr/bin/env python3
"""
Simple test script for FastAPI backend
Tests basic functionality without GPT-4 Vision dependency
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sqlite3
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Smart Meter Reading API - Test",
    description="Test version without GPT-4 Vision",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Meter Reading API - Test Mode",
        "version": "1.0.0",
        "status": "running",
        "mode": "test",
        "endpoints": {
            "health": "/health",
            "database": "/database/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpt_reader_available": False,
        "database_available": os.path.exists("../smart_meter_database.db"),
        "mode": "test"
    }

@app.get("/database/stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect("../smart_meter_database.db")
        cursor = conn.cursor()
        
        # Get total records
        cursor.execute("SELECT COUNT(*) FROM meter_readings")
        total_records = cursor.fetchone()[0]
        
        # Get date range
        cursor.execute("SELECT MIN(reading_date), MAX(reading_date) FROM meter_readings")
        date_range = cursor.fetchone()
        
        # Get unique meter serials
        cursor.execute("SELECT COUNT(DISTINCT meter_serial_number) FROM meter_readings")
        unique_meters = cursor.fetchone()[0]
        
        # Get reading statistics
        cursor.execute("SELECT MIN(meter_reading), MAX(meter_reading), AVG(meter_reading) FROM meter_readings")
        reading_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_records": total_records,
            "unique_meters": unique_meters,
            "date_range": {
                "earliest": date_range[0],
                "latest": date_range[1]
            },
            "reading_statistics": {
                "minimum": reading_stats[0],
                "maximum": reading_stats[1],
                "average": reading_stats[2]
            }
        }
        
    except Exception as e:
        return {
            "error": f"Database error: {str(e)}",
            "total_records": 0,
            "unique_meters": 0,
            "date_range": {"earliest": None, "latest": None},
            "reading_statistics": {"minimum": 0, "maximum": 0, "average": 0}
        }

@app.get("/database/search")
async def search_database(limit: int = 10):
    """Search database for records"""
    try:
        conn = sqlite3.connect("../smart_meter_database.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM meter_readings LIMIT ?", (limit,))
        results = cursor.fetchall()
        conn.close()
        
        # Format results
        formatted_results = []
        for row in results:
            formatted_results.append({
                "id": row[0],
                "meter_serial_number": row[1],
                "meter_reading": row[2],
                "original_reading": row[3],
                "reading_unit": row[4],
                "reading_date": row[5]
            })
        
        return {
            "query": {"limit": limit},
            "results": formatted_results,
            "count": len(formatted_results)
        }
        
    except Exception as e:
        return {
            "error": f"Database error: {str(e)}",
            "query": {"limit": limit},
            "results": [],
            "count": 0
        }

if __name__ == "__main__":
    print("🚀 Starting Smart Meter Reading API - Test Mode")
    print("📊 This version runs without GPT-4 Vision for testing")
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8000) 