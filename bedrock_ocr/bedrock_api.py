import os
import uuid
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from bedrock_ocr import BedrockOCRReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Smart Meter Bedrock OCR API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
bedrock_reader = None
batch_tasks = {}  # Store batch processing tasks
upload_dir = "bedrock_uploads"

# Ensure upload directory exists
os.makedirs(upload_dir, exist_ok=True)

# Pydantic models
class ProcessingResult(BaseModel):
    extracted_data: Dict[str, str]
    manual_data: Dict[str, str]
    validation: Dict[str, Any]
    ai_confidence: float
    extraction_notes: str
    image_quality: Dict[str, float]

class BatchStatus(BaseModel):
    batch_id: str
    status: str
    total_images: int
    processed: int
    successful: int
    failed: int
    progress_percentage: float

class BatchResult(BaseModel):
    batch_id: str
    total_images: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]

@app.on_event("startup")
async def startup_event():
    """Initialize Bedrock OCR reader on startup"""
    global bedrock_reader
    try:
        bedrock_reader = BedrockOCRReader()
        logger.info("Bedrock OCR reader initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock OCR reader: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "bedrock-ocr-api"}

@app.post("/process", response_model=ProcessingResult)
async def process_single_image(file: UploadFile = File(...)):
    """Process a single image"""
    try:
        # Validate file
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Only image files are supported")
        
        # Save uploaded file
        file_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image
        result = await bedrock_reader.process_image(file_path)
        
        # Save result to database
        bedrock_reader.save_processing_result(file_path, result)
        
        # Clean up file
        os.remove(file_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch/upload")
async def batch_upload(files: List[UploadFile] = File(...)):
    """Upload multiple images for batch processing"""
    try:
        if len(files) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 files allowed per batch")
        
        batch_id = str(uuid.uuid4())
        batch_dir = os.path.join(upload_dir, batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        
        saved_files = []
        for file in files:
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            file_path = os.path.join(batch_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
        
        # Initialize batch task
        batch_tasks[batch_id] = {
            "status": "uploaded",
            "total_images": len(saved_files),
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "file_paths": saved_files,
            "results": []
        }
        
        logger.info(f"Batch upload completed: {batch_id} with {len(saved_files)} files")
        
        return {"batch_id": batch_id, "total_files": len(saved_files)}
        
    except Exception as e:
        logger.error(f"Error in batch upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch/process/{batch_id}")
async def start_batch_processing(batch_id: str, background_tasks: BackgroundTasks):
    """Start batch processing"""
    try:
        if batch_id not in batch_tasks:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        batch_tasks[batch_id]["status"] = "processing"
        batch_tasks[batch_id]["processed"] = 0
        batch_tasks[batch_id]["successful"] = 0
        batch_tasks[batch_id]["failed"] = 0
        
        # Start background processing
        background_tasks.add_task(process_batch_background, batch_id)
        
        return {"batch_id": batch_id, "status": "processing_started"}
        
    except Exception as e:
        logger.error(f"Error starting batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_batch_background(batch_id: str):
    """Background task for batch processing"""
    try:
        batch_info = batch_tasks[batch_id]
        file_paths = batch_info["file_paths"]
        
        logger.info(f"Starting background processing for batch: {batch_id}")
        
        # Process images
        for i, file_path in enumerate(file_paths):
            try:
                batch_info["processed"] = i + 1
                
                result = await bedrock_reader.process_image(file_path)
                bedrock_reader.save_processing_result(file_path, result)
                
                batch_info["results"].append({
                    "image_path": file_path,
                    "result": result,
                    "status": "success"
                })
                batch_info["successful"] += 1
                
                logger.info(f"Processing completed: {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                batch_info["results"].append({
                    "image_path": file_path,
                    "result": None,
                    "status": "failed"
                })
                batch_info["failed"] += 1
        
        batch_info["status"] = "completed"
        logger.info(f"Batch processing completed: {batch_id}. Successful: {batch_info['successful']}, Failed: {batch_info['failed']}")
        
    except Exception as e:
        logger.error(f"Error in background batch processing: {e}")
        batch_tasks[batch_id]["status"] = "failed"

@app.get("/batch/status/{batch_id}", response_model=BatchStatus)
async def get_batch_status(batch_id: str):
    """Get batch processing status"""
    try:
        if batch_id not in batch_tasks:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        batch_info = batch_tasks[batch_id]
        progress = (batch_info["processed"] / batch_info["total_images"] * 100) if batch_info["total_images"] > 0 else 0
        
        return BatchStatus(
            batch_id=batch_id,
            status=batch_info["status"],
            total_images=batch_info["total_images"],
            processed=batch_info["processed"],
            successful=batch_info["successful"],
            failed=batch_info["failed"],
            progress_percentage=round(progress, 2)
        )
        
    except Exception as e:
        logger.error(f"Error getting batch status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/batch/results/{batch_id}", response_model=BatchResult)
async def get_batch_results(batch_id: str):
    """Get batch processing results"""
    try:
        if batch_id not in batch_tasks:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        batch_info = batch_tasks[batch_id]
        
        return BatchResult(
            batch_id=batch_id,
            total_images=batch_info["total_images"],
            successful=batch_info["successful"],
            failed=batch_info["failed"],
            results=batch_info["results"]
        )
        
    except Exception as e:
        logger.error(f"Error getting batch results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processing/history")
async def get_processing_history(
    limit: int = 20,
    offset: int = 0,
    outcome: Optional[str] = None
):
    """Get processing history"""
    try:
        history = bedrock_reader.get_processing_history(limit, offset, outcome)
        return history
        
    except Exception as e:
        logger.error(f"Error getting processing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processing/stats")
async def get_processing_stats():
    """Get processing statistics"""
    try:
        stats = bedrock_reader.get_processing_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting processing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processing/history/search")
async def search_processing_history(
    query: str,
    limit: int = 20,
    offset: int = 0
):
    """Search processing history"""
    try:
        # This would need to be implemented in the BedrockOCRReader class
        # For now, return empty results
        return {
            "history": [],
            "total_count": 0,
            "limit": limit,
            "offset": offset,
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Error searching processing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Use different port to avoid conflicts 