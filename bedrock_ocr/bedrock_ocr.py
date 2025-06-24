import os
import json
import logging
import sqlite3
import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import cv2
import numpy as np
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BedrockOCRReader:
    """
    Amazon Bedrock OCR Reader using Claude Vision models for smart meter reading
    """
    
    def __init__(self, model_name: str = "anthropic.claude-3-sonnet-20240229-v1:0", 
                 database_path: str = "smart_meter_database.db"):
        """
        Initialize Bedrock OCR Reader
        
        Args:
            model_name: Bedrock model name for Claude Vision
            database_path: Path to SQLite database
        """
        self.model_name = model_name
        self.database_path = database_path
        
        # Initialize Bedrock client
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name='us-east-1'  # Adjust as needed
            )
            logger.info(f"Initialized Bedrock client with model: {model_name}")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS credentials.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
        
        # Database connection manager
        self._db_lock = threading.Lock()
        
        # Load reference data
        self.reference_data = self._load_reference_data()
        logger.info(f"Loaded {len(self.reference_data)} reference meter records")
    
    def _get_db_connection(self):
        """Get a database connection with proper error handling"""
        try:
            conn = sqlite3.connect(self.database_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def _load_reference_data(self) -> Dict[str, Dict]:
        """Load reference meter data from database"""
        try:
            with self._db_lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT meter_serial_number, meter_type, current_reading, 
                           installation_date, last_reading_date
                    FROM meter_reference_data
                """)
                
                reference_data = {}
                for row in cursor.fetchall():
                    reference_data[row['meter_serial_number']] = {
                        'meter_serial_number': row['meter_serial_number'],
                        'meter_type': row['meter_type'],
                        'current_reading': row['current_reading'],
                        'installation_date': row['installation_date'],
                        'last_reading_date': row['last_reading_date']
                    }
                
                conn.close()
                return reference_data
                
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            return {}
    
    def _assess_image_quality(self, image_path: str) -> Dict[str, float]:
        """Assess image quality for confidence adjustment"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {'blur': 1.0, 'brightness': 0.0, 'contrast': 0.0}
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Blur detection using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_factor = min(1.0, blur_score / 100.0)  # Normalize
            
            # Brightness assessment
            brightness = np.mean(gray)
            brightness_factor = 1.0 - abs(brightness - 128) / 128.0
            
            # Contrast assessment
            contrast = np.std(gray)
            contrast_factor = min(1.0, contrast / 50.0)
            
            return {
                'blur': blur_factor,
                'brightness': brightness_factor,
                'contrast': contrast_factor
            }
            
        except Exception as e:
            logger.warning(f"Image quality assessment failed: {e}")
            return {'blur': 0.5, 'brightness': 0.5, 'contrast': 0.5}
    
    def _adjust_confidence(self, base_confidence: float, image_quality: Dict, 
                          validation_confidence: float) -> float:
        """Adjust confidence based on image quality and validation results"""
        # Image quality factor (average of blur, brightness, contrast)
        quality_factor = (image_quality['blur'] + image_quality['brightness'] + image_quality['contrast']) / 3.0
        
        # Validation confidence factor
        validation_factor = validation_confidence / 100.0
        
        # Combined adjustment
        adjusted_confidence = base_confidence * 0.4 + quality_factor * 0.3 + validation_factor * 0.3
        
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for Bedrock API"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def _create_prompt(self) -> str:
        """Create detailed prompt for Claude Vision"""
        return """You are an expert smart meter reading system. Analyze the provided image of a smart meter and extract the following information:

CRITICAL REQUIREMENTS:
1. METER SERIAL NUMBER: Look for a unique identifier (usually 8-12 digits, may include letters)
2. METER READING: Extract the current kWh reading (numeric value, may have decimal places)
3. METER TYPE: Identify the meter type (e.g., "Electric", "Gas", "Water", "Smart Meter")
4. READING DATE: Extract the date when the reading was taken (if visible)

EXTRACTION GUIDELINES:
- For meter serial numbers: Look for numbers/letters on the meter body, display, or labels
- For readings: Focus on the main display showing kWh consumption
- Handle leading zeros in serial numbers (preserve them exactly)
- If any information is unclear or not visible, indicate "Not visible" or "Unclear"

CONFIDENCE ASSESSMENT:
Rate your confidence on a scale of 1-10 for each extracted field based on:
- Image clarity and resolution
- Text visibility and contrast
- Completeness of information
- Potential for misinterpretation

OUTPUT FORMAT:
Provide your response in this exact JSON format:
{
    "meter_serial_number": "extracted_serial_or_not_visible",
    "meter_reading": "extracted_reading_or_not_visible", 
    "meter_type": "extracted_type_or_not_visible",
    "reading_date": "extracted_date_or_not_visible",
    "confidence_score": confidence_number_1_to_10,
    "extraction_notes": "brief_notes_about_extraction_quality"
}

IMPORTANT: Ensure the JSON is properly formatted and all fields are included."""
    
    def _find_best_match(self, extracted_serial: str) -> Tuple[Optional[Dict], str, float]:
        """
        Find best matching meter in reference data using hierarchical matching
        
        Returns:
            Tuple of (best_match, match_type, confidence)
        """
        if not extracted_serial or extracted_serial.lower() in ['not visible', 'unclear', '']:
            return None, "No Match", 0.0
        
        extracted_serial = extracted_serial.strip()
        
        # Strategy 1: Exact match
        if extracted_serial in self.reference_data:
            return self.reference_data[extracted_serial], "Exact Match", 100.0
        
        # Strategy 2: Partial match (extracted is subset of reference)
        for ref_serial, ref_data in self.reference_data.items():
            if extracted_serial in ref_serial or ref_serial in extracted_serial:
                # Calculate similarity score
                similarity = len(set(extracted_serial) & set(ref_serial)) / len(set(extracted_serial) | set(ref_serial))
                if similarity > 0.7:  # 70% similarity threshold
                    return ref_data, "Partial Match", similarity * 100
        
        # Strategy 3: Fuzzy match using character similarity
        best_match = None
        best_score = 0.0
        
        for ref_serial, ref_data in self.reference_data.items():
            # Simple character-based similarity
            common_chars = sum(1 for c in extracted_serial if c in ref_serial)
            total_chars = max(len(extracted_serial), len(ref_serial))
            similarity = common_chars / total_chars if total_chars > 0 else 0
            
            if similarity > best_score and similarity > 0.5:  # 50% threshold
                best_score = similarity
                best_match = ref_data
        
        if best_match:
            return best_match, "Fuzzy Match", best_score * 100
        
        return None, "No Match", 0.0
    
    def _validate_reading(self, extracted_reading: str, reference_reading: str) -> Tuple[bool, float]:
        """Validate extracted reading against reference"""
        try:
            if not extracted_reading or extracted_reading.lower() in ['not visible', 'unclear']:
                return False, 0.0
            
            # Convert to float for comparison
            extracted = float(extracted_reading.replace(',', ''))
            reference = float(reference_reading.replace(',', ''))
            
            # Check if extracted reading is reasonable (within 50% of reference)
            if reference > 0:
                ratio = extracted / reference
                if 0.5 <= ratio <= 1.5:
                    # Calculate confidence based on how close the reading is
                    confidence = max(0.0, 100.0 - abs(ratio - 1.0) * 50.0)
                    return True, confidence
            
            return False, 0.0
            
        except (ValueError, TypeError):
            return False, 0.0
    
    def _determine_processing_outcome(self, serial_match_type: str, serial_confidence: float, 
                                    reading_validated: bool, reading_confidence: float) -> str:
        """Determine overall processing outcome"""
        # Perfect Match: Both serial and reading match with high confidence
        if (serial_match_type in ["Exact Match", "Partial Match"] and serial_confidence >= 80.0 and
            reading_validated and reading_confidence >= 80.0):
            return "Perfect Match"
        
        # Good Match: Reading matches but serial doesn't (or vice versa with high confidence)
        if reading_validated and reading_confidence >= 70.0:
            return "Good Match"
        
        # Partial Match: Some confidence in either serial or reading
        if (serial_confidence >= 50.0 or reading_confidence >= 50.0):
            return "Partial Match"
        
        # No Match: Low confidence in both
        return "No Match"
    
    async def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image and return extraction results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extraction results and validation
        """
        try:
            # Assess image quality
            image_quality = self._assess_image_quality(image_path)
            
            # Encode image
            base64_image = self._encode_image(image_path)
            
            # Create prompt
            prompt = self._create_prompt()
            
            # Prepare request body for Bedrock
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            }
            
            # Make API call
            response = self.bedrock_client.invoke_model(
                modelId=self.model_name,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    extraction_result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                logger.warning(f"Raw response: {content}")
                # Fallback extraction
                extraction_result = {
                    "meter_serial_number": "Not visible",
                    "meter_reading": "Not visible",
                    "meter_type": "Not visible",
                    "reading_date": "Not visible",
                    "confidence_score": 1,
                    "extraction_notes": "Failed to parse response"
                }
            
            # Extract values with fallbacks
            extracted_serial = extraction_result.get("meter_serial_number", "Not visible")
            extracted_reading = extraction_result.get("meter_reading", "Not visible")
            extracted_type = extraction_result.get("meter_type", "Not visible")
            extracted_date = extraction_result.get("reading_date", "Not visible")
            base_confidence = float(extraction_result.get("confidence_score", 5)) / 10.0  # Convert 1-10 to 0-1
            extraction_notes = extraction_result.get("extraction_notes", "")
            
            # Find best match for serial number
            best_match, match_type, serial_confidence = self._find_best_match(extracted_serial)
            
            # Validate reading if we have a match
            reading_validated = False
            reading_confidence = 0.0
            if best_match and extracted_reading.lower() not in ['not visible', 'unclear']:
                reading_validated, reading_confidence = self._validate_reading(
                    extracted_reading, best_match['current_reading']
                )
            
            # Adjust confidence based on image quality and validation
            adjusted_confidence = self._adjust_confidence(
                base_confidence, image_quality, 
                max(serial_confidence, reading_confidence)
            )
            
            # Determine processing outcome
            processing_outcome = self._determine_processing_outcome(
                match_type, serial_confidence, reading_validated, reading_confidence
            )
            
            # Prepare result
            result = {
                "extracted_data": {
                    "meter_serial_number": extracted_serial,
                    "meter_reading": extracted_reading,
                    "meter_type": extracted_type,
                    "reading_date": extracted_date
                },
                "manual_data": {
                    "meter_serial_number": best_match['meter_serial_number'] if best_match else "No match found",
                    "meter_reading": best_match['current_reading'] if best_match else "No match found",
                    "meter_type": best_match['meter_type'] if best_match else "No match found",
                    "reading_date": best_match['last_reading_date'] if best_match else "No match found"
                },
                "validation": {
                    "serial_match_type": match_type,
                    "serial_confidence": serial_confidence,
                    "reading_validated": reading_validated,
                    "reading_confidence": reading_confidence,
                    "processing_outcome": processing_outcome
                },
                "ai_confidence": adjusted_confidence,
                "extraction_notes": extraction_notes,
                "image_quality": image_quality
            }
            
            logger.info(f"Successfully processed image: {image_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                "extracted_data": {
                    "meter_serial_number": "Error",
                    "meter_reading": "Error",
                    "meter_type": "Error",
                    "reading_date": "Error"
                },
                "manual_data": {
                    "meter_serial_number": "Error",
                    "meter_reading": "Error",
                    "meter_type": "Error",
                    "reading_date": "Error"
                },
                "validation": {
                    "serial_match_type": "Error",
                    "serial_confidence": 0.0,
                    "reading_validated": False,
                    "reading_confidence": 0.0,
                    "processing_outcome": "Error"
                },
                "ai_confidence": 0.0,
                "extraction_notes": f"Processing error: {str(e)}",
                "image_quality": {"blur": 0.0, "brightness": 0.0, "contrast": 0.0}
            }
    
    def save_processing_result(self, image_path: str, result: Dict[str, Any]) -> bool:
        """Save processing result to database"""
        try:
            with self._db_lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO processing_results (
                        image_path, extracted_serial, extracted_reading, extracted_type, extracted_date,
                        manual_serial, manual_reading, manual_type, manual_date,
                        serial_match_type, serial_confidence, reading_validated, reading_confidence,
                        processing_outcome, ai_confidence, extraction_notes, processing_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_path,
                    result['extracted_data']['meter_serial_number'],
                    result['extracted_data']['meter_reading'],
                    result['extracted_data']['meter_type'],
                    result['extracted_data']['reading_date'],
                    result['manual_data']['meter_serial_number'],
                    result['manual_data']['meter_reading'],
                    result['manual_data']['meter_type'],
                    result['manual_data']['reading_date'],
                    result['validation']['serial_match_type'],
                    result['validation']['serial_confidence'],
                    result['validation']['reading_validated'],
                    result['validation']['reading_confidence'],
                    result['validation']['processing_outcome'],
                    result['ai_confidence'],
                    result['extraction_notes'],
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Processing result saved to database for image: {image_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving processing result: {e}")
            return False
    
    async def process_batch(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary containing batch results
        """
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        results = []
        successful = 0
        failed = 0
        
        # Process images with concurrency control
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def process_single(image_path):
            async with semaphore:
                try:
                    result = await self.process_image(image_path)
                    if self.save_processing_result(image_path, result):
                        return {"image_path": image_path, "result": result, "status": "success"}
                    else:
                        return {"image_path": image_path, "result": None, "status": "failed"}
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    return {"image_path": image_path, "result": None, "status": "failed"}
        
        # Process all images concurrently
        tasks = [process_single(path) for path in image_paths]
        batch_results = await asyncio.gather(*tasks)
        
        # Count results
        for item in batch_results:
            if item['status'] == 'success':
                successful += 1
                results.append(item)
            else:
                failed += 1
        
        logger.info(f"Batch processing completed. Successful: {successful}, Failed: {failed}")
        
        return {
            "total_images": len(image_paths),
            "successful": successful,
            "failed": failed,
            "results": results
        }
    
    def get_processing_history(self, limit: int = 20, offset: int = 0, 
                             outcome_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get processing history from database"""
        try:
            with self._db_lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # Build query with optional filter
                query = """
                    SELECT * FROM processing_results 
                    WHERE 1=1
                """
                params = []
                
                if outcome_filter:
                    query += " AND processing_outcome = ?"
                    params.append(outcome_filter)
                
                query += " ORDER BY processing_timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                history = []
                for row in rows:
                    history.append({
                        'id': row['id'],
                        'image_path': row['image_path'],
                        'extracted_serial': row['extracted_serial'],
                        'extracted_reading': row['extracted_reading'],
                        'extracted_type': row['extracted_type'],
                        'extracted_date': row['extracted_date'],
                        'manual_serial': row['manual_serial'],
                        'manual_reading': row['manual_reading'],
                        'manual_type': row['manual_type'],
                        'manual_date': row['manual_date'],
                        'serial_match_type': row['serial_match_type'],
                        'serial_confidence': row['serial_confidence'],
                        'reading_validated': bool(row['reading_validated']),
                        'reading_confidence': row['reading_confidence'],
                        'processing_outcome': row['processing_outcome'],
                        'ai_confidence': row['ai_confidence'],
                        'extraction_notes': row['extraction_notes'],
                        'processing_timestamp': row['processing_timestamp']
                    })
                
                # Get total count
                count_query = "SELECT COUNT(*) FROM processing_results"
                if outcome_filter:
                    count_query += " WHERE processing_outcome = ?"
                    cursor.execute(count_query, [outcome_filter])
                else:
                    cursor.execute(count_query)
                
                total_count = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    "history": history,
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset
                }
                
        except Exception as e:
            logger.error(f"Error getting processing history: {e}")
            return {"history": [], "total_count": 0, "limit": limit, "offset": offset}
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            with self._db_lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # Total records
                cursor.execute("SELECT COUNT(*) FROM processing_results")
                total_records = cursor.fetchone()[0]
                
                # Outcome distribution
                cursor.execute("""
                    SELECT processing_outcome, COUNT(*) as count 
                    FROM processing_results 
                    GROUP BY processing_outcome
                """)
                outcome_stats = {row['processing_outcome']: row['count'] for row in cursor.fetchall()}
                
                # Average confidence
                cursor.execute("SELECT AVG(ai_confidence) FROM processing_results")
                avg_confidence = cursor.fetchone()[0] or 0.0
                
                # Recent activity (last 24 hours)
                cursor.execute("""
                    SELECT COUNT(*) FROM processing_results 
                    WHERE processing_timestamp >= datetime('now', '-1 day')
                """)
                recent_activity = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    "total_records": total_records,
                    "outcome_distribution": outcome_stats,
                    "average_confidence": round(avg_confidence, 2),
                    "recent_activity_24h": recent_activity
                }
                
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {
                "total_records": 0,
                "outcome_distribution": {},
                "average_confidence": 0.0,
                "recent_activity_24h": 0
            } 