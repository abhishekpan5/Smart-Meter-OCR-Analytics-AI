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
            "database_search": "/database/search",
            "processing_history": "/processing/history",
            "processing_stats": "/processing/stats",
            "processing_result": "/processing/result/{image_id}",
            "processing_search": "/processing/search",
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
    start_time = datetime.now()
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
        
        # Calculate processing duration
        end_time = datetime.now()
        processing_duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Save results to file
        results_file = RESULTS_DIR / f"{image_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "extraction_result": extraction_result,
                "validation_results": validation_results,
                "accuracy_report": accuracy_report,
                "processing_timestamp": datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        # Save results to database
        save_processing_result_to_db(
            image_id=image_id,
            file_info=file_info,
            extraction_result=extraction_result,
            validation_results=validation_results,
            accuracy_report=accuracy_report,
            processing_duration_ms=processing_duration_ms,
            results_file=str(results_file)
        )
        
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
        
        # Save error to database
        save_processing_result_to_db(
            image_id=image_id,
            file_info=file_info,
            extraction_result={},
            validation_results=[],
            accuracy_report={},
            processing_duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            results_file="",
            error=str(e)
        )

def save_processing_result_to_db(image_id: str, file_info: dict, extraction_result: dict, 
                                validation_results: list, accuracy_report: dict, 
                                processing_duration_ms: int, results_file: str, error: str = None):
    """Save processing results to the database"""
    try:
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()
        
        # Extract data from results
        extraction = extraction_result or {}
        validation = validation_results[0] if validation_results else {}
        
        # Get manual data from closest match if available
        manual_data = {}
        match_type = "no_match"
        match_confidence = 0.0
        
        if validation.get("closest_matches"):
            best_match = validation["closest_matches"][0]
            manual_data = {
                "manual_meter_serial": str(best_match.get("db_serial", "")),
                "manual_meter_reading": str(best_match.get("db_reading", "")),
                "manual_reading_date": str(best_match.get("db_date", "")),
                "manual_reading_unit": str(best_match.get("db_unit", ""))
            }
            
            # Extract match type and confidence from the match
            match_type = best_match.get("match_type", "no_match")
            
            # Calculate match confidence based on match type AND percentage error
            percentage_error = best_match.get("percentage_error", 100.0)
            
            if match_type == "exact_serial":
                if percentage_error == 0.0:
                    match_confidence = 1.0  # Perfect match
                elif percentage_error <= 1.0:
                    match_confidence = 0.95  # Very close match
                elif percentage_error <= 5.0:
                    match_confidence = 0.85  # Good match
                else:
                    match_confidence = 0.7  # Partial match
            elif match_type == "partial_serial":
                if percentage_error == 0.0:
                    match_confidence = 0.8  # Good partial match
                elif percentage_error <= 1.0:
                    match_confidence = 0.75  # Close partial match
                elif percentage_error <= 5.0:
                    match_confidence = 0.65  # Partial match
                else:
                    match_confidence = 0.5  # Poor partial match
            elif match_type == "fuzzy_serial":
                if percentage_error == 0.0:
                    match_confidence = 0.7  # Good fuzzy match
                elif percentage_error <= 1.0:
                    match_confidence = 0.65  # Close fuzzy match
                elif percentage_error <= 5.0:
                    match_confidence = 0.55  # Fuzzy match
                else:
                    match_confidence = 0.4  # Poor fuzzy match
            else:
                # For other match types, use percentage error only
                if percentage_error == 0.0:
                    match_confidence = 0.6  # Default good match
                elif percentage_error <= 1.0:
                    match_confidence = 0.55  # Default close match
                elif percentage_error <= 5.0:
                    match_confidence = 0.45  # Default match
                else:
                    match_confidence = 0.3  # Default poor match
        
        # Determine processing outcome based on validation status and matches
        if error:
            processing_outcome = "Error"
            validation_status = "failed"
        else:
            validation_status = validation.get("validation_status", "failed")
            
            if validation_status == "match" and validation.get("closest_matches"):
                # Determine outcome based on match type and confidence
                if match_type == "exact_serial" and match_confidence >= 0.95:
                    processing_outcome = "Perfect Match"
                elif match_type == "exact_serial" and match_confidence >= 0.8:
                    processing_outcome = "Good Match"
                elif match_type == "partial_serial" and match_confidence >= 0.7:
                    processing_outcome = "Good Match"
                elif match_type == "partial_serial" and match_confidence >= 0.5:
                    processing_outcome = "Partial Match"
                elif match_type == "fuzzy_serial" and match_confidence >= 0.6:
                    processing_outcome = "Partial Match"
                elif match_confidence >= 0.6:
                    processing_outcome = "Partial Match"
                else:
                    processing_outcome = "Poor Match"
            elif validation.get("closest_matches"):
                processing_outcome = "No Match"
            else:
                processing_outcome = "No Reference Data"
        
        # Handle AI confidence score - ensure it's between 0 and 1
        ai_confidence = extraction.get("confidence", 0.0)
        if isinstance(ai_confidence, (int, float)):
            # If it's a number like 9, convert to 0.9
            if ai_confidence > 1.0:
                ai_confidence = ai_confidence / 10.0
        elif isinstance(ai_confidence, str):
            # If it's a string like "9/10", convert to float
            if "/" in ai_confidence:
                try:
                    parts = ai_confidence.split("/")
                    ai_confidence = float(parts[0]) / float(parts[1])
                except:
                    ai_confidence = 0.0
            else:
                try:
                    ai_confidence = float(ai_confidence)
                    if ai_confidence > 1.0:
                        ai_confidence = ai_confidence / 10.0
                except:
                    ai_confidence = 0.0
        
        # Ensure confidence is between 0 and 1
        ai_confidence = max(0.0, min(1.0, ai_confidence))
        
        # Prepare data for insertion
        insert_data = {
            'image_id': image_id,
            'original_filename': file_info.get("original_filename", ""),
            'processing_timestamp': datetime.now().isoformat(),
            'extracted_meter_serial': str(extraction.get("meter_serial", "")),
            'extracted_meter_reading': str(extraction.get("meter_reading", "")),
            'extracted_reading_date': str(extraction.get("date", "")),
            'extracted_reading_time': str(extraction.get("time", "")),
            'ai_confidence_score': ai_confidence,
            'manual_meter_serial': manual_data.get("manual_meter_serial", ""),
            'manual_meter_reading': manual_data.get("manual_meter_reading", ""),
            'manual_reading_date': manual_data.get("manual_reading_date", ""),
            'manual_reading_unit': manual_data.get("manual_reading_unit", ""),
            'processing_outcome': processing_outcome,
            'match_type': match_type,
            'match_confidence': match_confidence,
            'validation_status': validation_status,
            'closest_matches_count': len(validation.get("closest_matches", [])),
            'processing_duration_ms': processing_duration_ms,
            'gpt_model_used': 'gpt-4-vision-preview',
            'image_file_path': str(file_info.get("file_path", "")),
            'result_file_path': results_file
        }
        
        # Insert into database
        insert_query = """
        INSERT INTO processing_results (
            image_id, original_filename, processing_timestamp,
            extracted_meter_serial, extracted_meter_reading, extracted_reading_date, extracted_reading_time, ai_confidence_score,
            manual_meter_serial, manual_meter_reading, manual_reading_date, manual_reading_unit,
            processing_outcome, match_type, match_confidence, validation_status, closest_matches_count,
            processing_duration_ms, gpt_model_used, image_file_path, result_file_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.execute(insert_query, (
            insert_data['image_id'], insert_data['original_filename'], insert_data['processing_timestamp'],
            insert_data['extracted_meter_serial'], insert_data['extracted_meter_reading'], 
            insert_data['extracted_reading_date'], insert_data['extracted_reading_time'], insert_data['ai_confidence_score'],
            insert_data['manual_meter_serial'], insert_data['manual_meter_reading'], 
            insert_data['manual_reading_date'], insert_data['manual_reading_unit'],
            insert_data['processing_outcome'], insert_data['match_type'], insert_data['match_confidence'],
            insert_data['validation_status'], insert_data['closest_matches_count'],
            insert_data['processing_duration_ms'], insert_data['gpt_model_used'],
            insert_data['image_file_path'], insert_data['result_file_path']
        ))
        
        conn.commit()
        logger.info(f"Processing results saved to database for image: {image_id}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error saving processing results to database: {e}")

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

@app.get("/processing/history")
async def get_processing_history(limit: int = 50, offset: int = 0, outcome: Optional[str] = None):
    """Get processing history from the database"""
    try:
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()
        
        # Build query with optional filtering
        query = """
            SELECT * FROM processing_results 
        """
        params = []
        
        if outcome:
            query += " WHERE processing_outcome = ?"
            params.append(outcome)
        
        query += " ORDER BY processing_timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Get column names
        cursor.execute("PRAGMA table_info(processing_results)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Format results
        formatted_results = []
        for row in results:
            result_dict = dict(zip(columns, row))
            formatted_results.append(result_dict)
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM processing_results"
        if outcome:
            count_query += " WHERE processing_outcome = ?"
            cursor.execute(count_query, [outcome])
        else:
            cursor.execute(count_query)
        
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "results": formatted_results,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total_count
        }
        
    except Exception as e:
        logger.error(f"Processing history error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/processing/stats")
async def get_processing_stats():
    """Get processing statistics from the database"""
    try:
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()
        
        # Total processing attempts
        cursor.execute("SELECT COUNT(*) FROM processing_results")
        total_attempts = cursor.fetchone()[0]
        
        # Outcomes breakdown
        cursor.execute("""
            SELECT processing_outcome, COUNT(*) as count 
            FROM processing_results 
            GROUP BY processing_outcome
        """)
        outcomes = dict(cursor.fetchall())
        
        # Average processing time
        cursor.execute("SELECT AVG(processing_duration_ms) FROM processing_results WHERE processing_duration_ms > 0")
        avg_processing_time = cursor.fetchone()[0] or 0
        
        # Average AI confidence
        cursor.execute("SELECT AVG(ai_confidence_score) FROM processing_results WHERE ai_confidence_score > 0")
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Recent activity (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM processing_results 
            WHERE datetime(processing_timestamp) >= datetime('now', '-1 day')
        """)
        recent_activity = cursor.fetchone()[0]
        
        # Success rate
        success_count = outcomes.get("Perfect Match", 0) + outcomes.get("Good Match", 0) + outcomes.get("Partial Match", 0)
        success_rate = (success_count / total_attempts * 100) if total_attempts > 0 else 0
        
        conn.close()
        
        return {
            "total_attempts": total_attempts,
            "outcomes": outcomes,
            "success_rate": round(success_rate, 2),
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "avg_ai_confidence": round(avg_confidence, 3),
            "recent_activity_24h": recent_activity
        }
        
    except Exception as e:
        logger.error(f"Processing stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/processing/result/{image_id}")
async def get_processing_result_from_db(image_id: str):
    """Get processing result from database by image ID"""
    try:
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM processing_results WHERE image_id = ?", (image_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Processing result not found")
        
        # Get column names
        cursor.execute("PRAGMA table_info(processing_results)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Format result
        result_dict = dict(zip(columns, result))
        
        conn.close()
        
        return result_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing result error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/processing/search")
async def search_processing_results(
    serial: Optional[str] = None, 
    outcome: Optional[str] = None, 
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 20
):
    """Search processing results with various filters"""
    try:
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()
        
        # Build query with filters
        query = "SELECT * FROM processing_results WHERE 1=1"
        params = []
        
        if serial:
            query += " AND (extracted_meter_serial LIKE ? OR manual_meter_serial LIKE ?)"
            params.extend([f"%{serial}%", f"%{serial}%"])
        
        if outcome:
            query += " AND processing_outcome = ?"
            params.append(outcome)
        
        if date_from:
            query += " AND datetime(processing_timestamp) >= datetime(?)"
            params.append(date_from)
        
        if date_to:
            query += " AND datetime(processing_timestamp) <= datetime(?)"
            params.append(date_to)
        
        query += " ORDER BY processing_timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Get column names
        cursor.execute("PRAGMA table_info(processing_results)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Format results
        formatted_results = []
        for row in results:
            result_dict = dict(zip(columns, row))
            formatted_results.append(result_dict)
        
        conn.close()
        
        return {
            "query": {
                "serial": serial,
                "outcome": outcome,
                "date_from": date_from,
                "date_to": date_to,
                "limit": limit
            },
            "results": formatted_results,
            "count": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Processing search error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 