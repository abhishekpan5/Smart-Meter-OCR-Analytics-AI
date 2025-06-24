from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import uuid
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
import sys
import shutil
import base64
from PIL import Image
import io

# Add parent directory to path to import GPT OCR module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gpt_ocr import GPT4VisionMeterReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Meter Reading API",
    description="AI-powered smart meter reading with GPT-4 Vision and database validation",
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

# Pydantic models
class ProcessingRequest(BaseModel):
    image_id: str
    detail_level: Optional[str] = "high"
    enable_validation: Optional[bool] = True

class ProcessingResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ValidationResult(BaseModel):
    image_path: str
    gpt4_reading: str
    gpt4_confidence: float
    gpt4_date: Optional[str]
    gpt4_time: Optional[str]
    database_matches: List[tuple]
    validation_status: str
    accuracy_metrics: Dict[str, Any]
    processing_timestamp: str

# Global variables
UPLOAD_DIR = Path("smart_meter_ui/backend/uploads")
PROCESSED_DIR = Path("smart_meter_ui/backend/processed")
RESULTS_DIR = Path("smart_meter_ui/backend/results")
DATABASE_PATH = Path("../../smart_meter_database.db")  # Path to database from backend directory

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize GPT-4 Vision reader
try:
    meter_reader = GPT4VisionMeterReader(str(DATABASE_PATH))
    logger.info("GPT-4 Vision reader initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize GPT-4 Vision reader: {e}")
    meter_reader = None

# Processing queue
processing_queue = {}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Meter Reading API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "/upload",
            "process": "/process/{image_id}",
            "status": "/status/{image_id}",
            "results": "/results/{image_id}",
            "database": "/database/stats",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpt_reader_available": meter_reader is not None,
        "database_available": os.path.exists(str(DATABASE_PATH))
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for processing"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique ID for the image
        image_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{image_id}{file_extension}"
        file_path = UPLOAD_DIR / filename
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Store file info
        processing_queue[image_id] = {
            "status": "uploaded",
            "filename": filename,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "upload_time": datetime.now().isoformat(),
            "file_size": os.path.getsize(file_path)
        }
        
        logger.info(f"Image uploaded successfully: {image_id}")
        
        return {
            "status": "success",
            "message": "Image uploaded successfully",
            "image_id": image_id,
            "filename": filename,
            "file_size": processing_queue[image_id]["file_size"]
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/process/{image_id}")
async def process_image(image_id: str, background_tasks: BackgroundTasks):
    """Process an uploaded image with GPT-4 Vision"""
    try:
        if image_id not in processing_queue:
            raise HTTPException(status_code=404, detail="Image not found")
        
        if meter_reader is None:
            raise HTTPException(status_code=500, detail="GPT-4 Vision reader not available")
        
        # Update status to processing
        processing_queue[image_id]["status"] = "processing"
        processing_queue[image_id]["process_start_time"] = datetime.now().isoformat()
        
        # Start background processing
        background_tasks.add_task(process_image_background, image_id)
        
        return {
            "status": "success",
            "message": "Processing started",
            "image_id": image_id,
            "status_url": f"/status/{image_id}"
        }
        
    except Exception as e:
        logger.error(f"Process error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

async def process_image_background(image_id: str):
    """Background task to process image"""
    try:
        file_info = processing_queue[image_id]
        file_path = file_info["file_path"]
        
        logger.info(f"Processing image: {image_id}")
        
        # Process with GPT-4 Vision
        extraction_result = meter_reader.extract_meter_data(file_path)
        
        # Validate against database
        validation_results = meter_reader.validate_against_database([extraction_result])
        
        # Generate accuracy report
        accuracy_report = meter_reader.generate_enhanced_accuracy_report(validation_results)
        
        # Save results
        results_file = RESULTS_DIR / f"{image_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "extraction_result": extraction_result,
                "validation_results": validation_results,
                "accuracy_report": accuracy_report,
                "processing_timestamp": datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        # Update processing queue
        processing_queue[image_id].update({
            "status": "completed",
            "process_end_time": datetime.now().isoformat(),
            "results_file": str(results_file),
            "extraction_result": extraction_result,
            "validation_results": validation_results,
            "accuracy_report": accuracy_report
        })
        
        logger.info(f"Processing completed: {image_id}")
        
    except Exception as e:
        logger.error(f"Background processing error for {image_id}: {e}")
        processing_queue[image_id].update({
            "status": "failed",
            "error": str(e),
            "process_end_time": datetime.now().isoformat()
        })

@app.get("/status/{image_id}")
async def get_processing_status(image_id: str):
    """Get processing status for an image"""
    if image_id not in processing_queue:
        raise HTTPException(status_code=404, detail="Image not found")
    
    file_info = processing_queue[image_id]
    
    return {
        "image_id": image_id,
        "status": file_info["status"],
        "filename": file_info["filename"],
        "original_filename": file_info["original_filename"],
        "upload_time": file_info["upload_time"],
        "file_size": file_info["file_size"],
        "process_start_time": file_info.get("process_start_time"),
        "process_end_time": file_info.get("process_end_time"),
        "error": file_info.get("error")
    }

@app.get("/results/{image_id}")
async def get_processing_results(image_id: str):
    """Get detailed processing results for an image"""
    if image_id not in processing_queue:
        raise HTTPException(status_code=404, detail="Image not found")
    
    file_info = processing_queue[image_id]
    
    if file_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Processing not completed. Status: {file_info['status']}")
    
    # Load results from file
    results_file = Path(file_info["results_file"])
    if not results_file.exists():
        raise HTTPException(status_code=500, detail="Results file not found")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Format response for frontend
    extraction = results["extraction_result"]
    validation = results["validation_results"][0] if results["validation_results"] else {}
    accuracy = results["accuracy_report"]
    
    response = {
        "image_info": {
            "image_id": image_id,
            "filename": file_info["filename"],
            "original_filename": file_info["original_filename"],
            "upload_time": file_info["upload_time"],
            "processing_time": file_info["process_end_time"]
        },
        "extraction_results": {
            "meter_reading": extraction.get("meter_reading"),
            "meter_type": extraction.get("meter_type"),
            "meter_serial": extraction.get("meter_serial"),
            "units": extraction.get("units"),
            "display_type": extraction.get("display_type"),
            "date": extraction.get("date"),
            "time": extraction.get("time"),
            "confidence": extraction.get("confidence"),
            "date_time_confidence": extraction.get("date_time_confidence"),
            "additional_text": extraction.get("additional_text"),
            "extraction_notes": extraction.get("extraction_notes")
        },
        "validation_results": {
            "validation_status": validation.get("validation_status"),
            "database_matches": validation.get("database_matches", []),
            "closest_matches": validation.get("closest_matches", []),
            "temporal_validation": validation.get("temporal_validation", {})
        },
        "accuracy_metrics": {
            "processing_summary": accuracy.get("processing_summary", {}),
            "accuracy_metrics": accuracy.get("accuracy_metrics", {}),
            "temporal_analysis": accuracy.get("temporal_analysis", {}),
            "error_analysis": accuracy.get("error_analysis", {}),
            "recommendations": accuracy.get("recommendations", [])
        },
        "raw_data": {
            "extraction_result": extraction,
            "validation_results": results["validation_results"],
            "accuracy_report": accuracy
        }
    }
    
    return response

@app.get("/database/stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect(str(DATABASE_PATH))
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
        logger.error(f"Database stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/database/search")
async def search_database(reading: Optional[str] = None, serial: Optional[str] = None, limit: int = 10):
    """Search database for specific readings or serial numbers"""
    try:
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()
        
        if reading:
            cursor.execute("""
                SELECT * FROM meter_readings 
                WHERE meter_reading = ?
                LIMIT ?
            """, (reading, limit))
        elif serial:
            cursor.execute("""
                SELECT * FROM meter_readings 
                WHERE meter_serial_number = ?
                LIMIT ?
            """, (serial, limit))
        else:
            cursor.execute("SELECT * FROM meter_readings LIMIT ?", (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Format results
        formatted_results = []
        for row in results:
            formatted_results.append({
                "meter_serial_number": row[0],
                "meter_reading": row[1],
                "reading_unit": row[2],
                "reading_date": row[3]
            })
        
        return {
            "query": {"reading": reading, "serial": serial, "limit": limit},
            "results": formatted_results,
            "count": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Database search error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 