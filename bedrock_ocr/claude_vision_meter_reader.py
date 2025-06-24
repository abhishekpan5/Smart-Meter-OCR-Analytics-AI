import os
import base64
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import re
import io
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeVisionMeterReader:
    """
    Advanced Smart Meter Reading using Amazon Bedrock Claude Vision models
    Integrates with existing SQLite database for validation
    Supports batch inference for parallel processing
    """
    
    def __init__(self, db_path: str = None, region_name: str = None):
        """
        Initialize Claude Vision client with Amazon Bedrock
        
        Args:
            db_path: Path to SQLite database (defaults to environment variable)
            region_name: AWS region name (defaults to environment variable)
        """
        # Load AWS credentials and configuration from environment variables
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("AWS credentials not found in environment variables. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env file.")
        
        # Initialize Bedrock client
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
            logger.info(f"Bedrock client initialized successfully in region: {self.region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
        
        # Load database path from environment or use default
        self.db_path = db_path or os.getenv("DATABASE_PATH", "smart_meter_database.db")
        
        # Load configuration from environment variables
        self.model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.detail_level = os.getenv("DETAIL_LEVEL", "high")
        
        # Batch processing configuration
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "5"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "60"))
        
        # Enhanced validation patterns including date/time
        self.reading_patterns = [
            r'\b\d{4,6}\b',  # 4-6 digit readings
            r'\d+\.\d+',     # Decimal readings
            r'\d{5}',        # 5-digit readings (most common)
        ]
        
        # Date/time patterns for validation
        self.datetime_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # Date formats: DD/MM/YYYY, MM-DD-YY
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',    # ISO date format: YYYY-MM-DD
            r'\d{1,2}:\d{2}(?::\d{2})?',       # Time formats: HH:MM or HH:MM:SS
            r'\d{1,2}\s*[AP]M',                # AM/PM format
        ]
        
        logger.info(f"Initialized Claude Vision Reader with model: {self.model_id}")
        logger.info(f"Database path: {self.db_path}")
        logger.info(f"Batch processing: max {self.max_concurrent_requests} concurrent requests, batch size {self.batch_size}")
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string for Bedrock API
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise
    
    def create_enhanced_prompt(self) -> str:
        """
        Create a detailed prompt for meter reading and date/time extraction with enhanced confidence criteria
        
        Returns:
            Enhanced structured prompt for Claude Vision
        """
        # Load custom prompt from environment if available
        custom_prompt = os.getenv("CUSTOM_PROMPT")
        if custom_prompt:
            return custom_prompt
        
        return """
        You are an expert in reading digital utility meters and extracting temporal information. Analyze this meter image and extract the following information:

        1. METER READING: The main numerical reading displayed (usually 4-6 digits)
        2. METER TYPE: Type of meter (electricity, gas, water, etc.)
        3. METER SERIAL/ID: Any visible serial number or meter ID
        4. UNITS: Unit of measurement (kWh, cubic meters, etc.)
        5. DISPLAY TYPE: LCD, LED, analog, or digital
        6. DATE: Any date information visible on the display (format: YYYY-MM-DD if possible)
        7. TIME: Any time information visible on the display (format: HH:MM:SS if possible)
        8. TIMESTAMP_FORMAT: The format of the date/time as it appears on the meter
        9. ADDITIONAL TEXT: Any other visible text or markings
        10. IMAGE_QUALITY_FACTORS: Assess the following factors (1-10 scale each):
            - CLARITY: How sharp and clear is the image?
            - LIGHTING: How well-lit is the meter display?
            - ANGLE: How straight-on is the viewing angle?
            - FOCUS: How well-focused is the meter display?
            - CONTRAST: How much contrast between digits and background?
            - GLARE: How much glare or reflection is present?
            - OCCLUSION: How much of the display is blocked or obscured?
        11. CONFIDENCE: Your confidence level (1-10) in the reading accuracy based on:
            - 10: Perfect clarity, all digits clearly visible, no ambiguity
            - 9: Excellent clarity, minor uncertainty in one digit
            - 8: Very good clarity, slight blur but readable
            - 7: Good clarity, some digits slightly unclear
            - 6: Moderate clarity, some digits require interpretation
            - 5: Fair clarity, significant uncertainty in multiple digits
            - 4: Poor clarity, many digits unclear or ambiguous
            - 3: Very poor clarity, most digits unclear
            - 2: Extremely poor clarity, barely readable
            - 1: Unreadable or completely unclear
        12. DATE_TIME_CONFIDENCE: Your confidence level (1-10) specifically for date/time extraction
        13. UNCERTAINTY_FACTORS: List any specific factors that reduce confidence:
            - Blurry digits
            - Poor lighting
            - Glare or reflections
            - Unusual angle
            - Partial occlusion
            - Similar-looking digits (6/8, 0/O, 1/l, etc.)
            - Display damage or wear
            - Multiple possible readings

        SPECIAL INSTRUCTIONS FOR DATE/TIME EXTRACTION:
        - Look carefully for any digital display showing date/time information
        - Common locations: top/bottom of LCD display, separate date/time panel
        - Pay attention to timestamp watermarks or overlay text
        - Note if date/time appears to be from the meter itself or from a camera/phone
        - If multiple timestamps are visible, extract all of them with their sources

        CONFIDENCE ASSESSMENT CRITERIA:
        - Consider image quality factors when assigning confidence
        - Be conservative in confidence assessment
        - If any digits are unclear or ambiguous, reduce confidence accordingly
        - Consider the difficulty of the specific meter type and display
        - Account for potential digit confusion (6/8, 0/O, 1/l, etc.)

        Please respond in the following JSON format:
        {
            "meter_reading": "extracted_reading",
            "meter_type": "electricity/gas/water/other",
            "meter_serial": "serial_if_visible",
            "units": "kWh/m3/other",
            "display_type": "LCD/LED/analog/digital",
            "date": "YYYY-MM-DD or original format",
            "time": "HH:MM:SS or original format",
            "timestamp_format": "description of how date/time appears",
            "timestamp_source": "meter_display/camera_overlay/watermark/other",
            "additional_timestamps": [
                {
                    "datetime": "additional timestamp if found",
                    "source": "where this timestamp is located",
                    "format": "format description"
                }
            ],
            "additional_text": "any_other_text",
            "image_quality_factors": {
                "clarity": clarity_score_1_10,
                "lighting": lighting_score_1_10,
                "angle": angle_score_1_10,
                "focus": focus_score_1_10,
                "contrast": contrast_score_1_10,
                "glare": glare_score_1_10,
                "occlusion": occlusion_score_1_10
            },
            "confidence": confidence_score_for_reading,
            "date_time_confidence": confidence_score_for_datetime,
            "uncertainty_factors": ["list", "of", "specific", "uncertainty", "factors"],
            "extraction_notes": "any_observations_or_uncertainties",
            "temporal_analysis": "description of all date/time elements found"
        }

        Focus on accuracy and be explicit about any uncertainty in your reading or date/time extraction.
        If no date/time information is visible, explicitly state this in the response.
        Be realistic about confidence levels - it's better to be conservative than overconfident.
        """
    
    def create_bedrock_request(self, base64_image: str, prompt: str) -> Dict[str, Any]:
        """
        Create Bedrock API request payload
        
        Args:
            base64_image: Base64 encoded image
            prompt: Text prompt for the model
            
        Returns:
            Bedrock request payload
        """
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
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
    
    def extract_meter_data(self, image_path: str, detail_level: str = None) -> Dict:
        """
        Extract meter reading and date/time using Claude Vision
        
        Args:
            image_path: Path to meter image
            detail_level: "low", "high", or "auto" (overrides environment default)
            
        Returns:
            Dictionary containing extracted meter data including date/time
        """
        try:
            # Use provided detail level or environment default
            detail = detail_level or self.detail_level
            
            # Encode image
            base64_image = self.encode_image(image_path)
            
            # Create prompt
            prompt = self.create_enhanced_prompt()
            
            # Create API request
            request_body = self.create_bedrock_request(base64_image, prompt)
            
            # Make API call
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response.get('body').read())
            response_text = response_body['content'][0]['text']
            
            # Try to parse JSON response
            try:
                # First, try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    meter_data = json.loads(json_str)
                else:
                    # Try direct JSON parsing
                    meter_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: extract reading and date/time from text response
                logger.warning("Failed to parse JSON response, attempting text extraction")
                meter_data = self._enhanced_fallback_extraction(response_text)
            
            # Validate and normalize date/time data
            meter_data = self._validate_datetime_data(meter_data)
            
            # Post-process confidence
            meter_data = self._post_process_confidence(meter_data)
            
            # Add metadata
            meter_data.update({
                "image_path": image_path,
                "extraction_timestamp": datetime.now().isoformat(),
                "api_model": self.model_id,
                "processing_tokens": response_body.get('usage', {}).get('input_tokens', 0),
                "raw_response": response_text,
                "configuration": {
                    "detail_level": detail,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "model_id": self.model_id
                }
            })
            
            return meter_data
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Bedrock API error ({error_code}): {error_message}")
            return {
                "error": f"Bedrock API error: {error_code} - {error_message}",
                "image_path": image_path,
                "extraction_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error extracting meter data from {image_path}: {e}")
            return {
                "error": str(e),
                "image_path": image_path,
                "extraction_timestamp": datetime.now().isoformat()
            }
    
    async def extract_meter_data_async(self, image_path: str, detail_level: str = None) -> Dict:
        """
        Asynchronous version of extract_meter_data for batch processing
        
        Args:
            image_path: Path to meter image
            detail_level: "low", "high", or "auto"
            
        Returns:
            Dictionary containing extracted meter data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_meter_data, image_path, detail_level)
    
    async def process_batch_async(self, image_paths: List[str], detail_level: str = None) -> List[Dict]:
        """
        Process multiple images asynchronously with concurrency control
        
        Args:
            image_paths: List of image file paths
            detail_level: "low", "high", or "auto"
            
        Returns:
            List of extraction results
        """
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def process_single_image(image_path: str) -> Dict:
            async with semaphore:
                try:
                    result = await self.extract_meter_data_async(image_path, detail_level)
                    logger.info(f"Processed: {image_path}")
                    return result
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    return {
                        "error": str(e),
                        "image_path": image_path,
                        "extraction_timestamp": datetime.now().isoformat()
                    }
        
        # Process images in batches
        results = []
        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}: {len(batch)} images")
            
            # Process batch concurrently
            batch_tasks = [process_single_image(path) for path in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Exception in batch processing: {result}")
                    results.append({
                        "error": str(result),
                        "image_path": batch[j],
                        "extraction_timestamp": datetime.now().isoformat()
                    })
                else:
                    results.append(result)
        
        logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def process_batch_sync(self, image_paths: List[str], detail_level: str = None) -> List[Dict]:
        """
        Synchronous batch processing using ThreadPoolExecutor
        
        Args:
            image_paths: List of image file paths
            detail_level: "low", "high", or "auto"
            
        Returns:
            List of extraction results
        """
        logger.info(f"Starting synchronous batch processing of {len(image_paths)} images")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.extract_meter_data, path, detail_level): path 
                for path in image_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result(timeout=self.request_timeout)
                    results.append(result)
                    logger.info(f"Processed: {path}")
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    results.append({
                        "error": str(e),
                        "image_path": path,
                        "extraction_timestamp": datetime.now().isoformat()
                    })
        
        logger.info(f"Synchronous batch processing completed: {len(results)} results")
        return results
    
    def _enhanced_fallback_extraction(self, response_text: str) -> Dict:
        """
        Enhanced fallback method to extract reading and date/time from unstructured response
        
        Args:
            response_text: Raw text response from API
            
        Returns:
            Dictionary with extracted data including date/time
        """
        reading = None
        date_found = None
        time_found = None
        meter_serial = None
        meter_type = None
        units = None
        display_type = None
        
        # Try to extract numerical reading
        for pattern in self.reading_patterns:
            matches = re.findall(pattern, response_text)
            if matches:
                reading = matches[0]  # Take first match
                break
        
        # Try to extract date/time information
        for pattern in self.datetime_patterns:
            matches = re.findall(pattern, response_text)
            if matches:
                if ':' in matches[0]:  # Likely a time
                    time_found = matches[0]
                else:  # Likely a date
                    date_found = matches[0]
        
        # Try to extract meter serial (look for patterns like "01816777" or "serial: 12345")
        serial_patterns = [
            r'"meter_serial":\s*"([^"]+)"',
            r'serial[:\s]+([A-Z0-9]{6,})',
            r'ID[:\s]+([A-Z0-9]{6,})',
            r'([A-Z0-9]{6,})',  # Any 6+ digit alphanumeric sequence
        ]
        for pattern in serial_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                meter_serial = match.group(1)
                break
        
        # Try to extract meter type
        type_patterns = [
            r'"meter_type":\s*"([^"]+)"',
            r'type[:\s]+(electricity|gas|water|electric)',
            r'(electricity|gas|water|electric)\s+meter',
        ]
        for pattern in type_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                meter_type = match.group(1)
                break
        
        # Try to extract units
        unit_patterns = [
            r'"units":\s*"([^"]+)"',
            r'units[:\s]+(kWh|m3|gal|l)',
            r'(kWh|m3|gal|l)\b',
        ]
        for pattern in unit_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                units = match.group(1)
                break
        
        # Try to extract display type
        display_patterns = [
            r'"display_type":\s*"([^"]+)"',
            r'display[:\s]+(LCD|LED|analog|digital)',
            r'(LCD|LED|analog|digital)\s+display',
        ]
        for pattern in display_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                display_type = match.group(1)
                break
        
        return {
            "meter_reading": reading,
            "meter_type": meter_type or "unknown",
            "meter_serial": meter_serial,
            "units": units,
            "display_type": display_type,
            "date": date_found,
            "time": time_found,
            "timestamp_source": "fallback_extraction",
            "image_quality_factors": {
                "clarity": 3,  # Low confidence for fallback
                "lighting": 3,
                "angle": 3,
                "focus": 3,
                "contrast": 3,
                "glare": 3,
                "occlusion": 3
            },
            "confidence": 3,  # Lower confidence for fallback
            "date_time_confidence": 2,  # Even lower confidence for fallback date/time
            "uncertainty_factors": ["JSON parsing failed", "fallback extraction used"],
            "extraction_notes": "Fallback extraction used due to JSON parsing failure",
            "raw_response": response_text
        }
    
    def _validate_datetime_data(self, meter_data: Dict) -> Dict:
        """
        Validate and normalize extracted date/time data
        
        Args:
            meter_data: Dictionary containing extracted meter data
            
        Returns:
            Validated meter data
        """
        # Validate date format
        if meter_data.get("date"):
            date_str = str(meter_data["date"])
            # Try to normalize date format
            try:
                # Add basic date validation here if needed
                pass
            except:
                logger.warning(f"Invalid date format: {date_str}")
        
        # Validate time format
        if meter_data.get("time"):
            time_str = str(meter_data["time"])
            # Try to normalize time format
            try:
                # Add basic time validation here if needed
                pass
            except:
                logger.warning(f"Invalid time format: {time_str}")
        
        return meter_data
    
    def _post_process_confidence(self, meter_data: Dict) -> Dict:
        """
        Post-process confidence scores based on image quality factors
        
        Args:
            meter_data: Dictionary containing extracted meter data
            
        Returns:
            Post-processed meter data
        """
        # Get image quality factors
        quality_factors = meter_data.get("image_quality_factors", {})
        
        if quality_factors:
            # Calculate average quality score
            quality_scores = [
                quality_factors.get("clarity", 5),
                quality_factors.get("lighting", 5),
                quality_factors.get("angle", 5),
                quality_factors.get("focus", 5),
                quality_factors.get("contrast", 5),
                quality_factors.get("glare", 5),
                quality_factors.get("occlusion", 5)
            ]
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            # Adjust confidence based on quality
            original_confidence = meter_data.get("confidence", 5)
            
            # Quality adjustment factor (0.8 to 1.2 range)
            quality_factor = 0.8 + (avg_quality / 10) * 0.4
            
            # Apply quality adjustment
            adjusted_confidence = min(10, max(1, original_confidence * quality_factor))
            
            meter_data["confidence"] = round(adjusted_confidence, 2)
            meter_data["quality_adjusted_confidence"] = True
            meter_data["average_quality_score"] = round(avg_quality, 2)
        
        return meter_data
    
    def validate_against_database(self, extraction_results: List[Dict]) -> List[Dict]:
        """
        Validate extracted results against the reference database
        
        Args:
            extraction_results: List of extraction results
            
        Returns:
            List of validation results
        """
        validation_results = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for result in extraction_results:
                if "error" in result:
                    validation_results.append({
                        "validation_status": "error",
                        "error": result["error"],
                        "extraction_result": result
                    })
                    continue
                
                # Extract data for validation
                extracted_reading = result.get("meter_reading", "")
                extracted_serial = result.get("meter_serial", "")
                extracted_date = result.get("date", "")
                
                # Perform fuzzy matching
                closest_matches = self._enhanced_fuzzy_matching(
                    cursor, extracted_reading, extracted_serial, extracted_date, 10.0
                )
                
                # Determine validation status
                validation_status = "no_match"
                if closest_matches:
                    best_match = closest_matches[0]
                    if best_match[4] <= 1.0:  # Percentage error <= 1%
                        validation_status = "exact_match"
                    elif best_match[4] <= 5.0:  # Percentage error <= 5%
                        validation_status = "close_match"
                    else:
                        validation_status = "fuzzy_match"
                
                # Create validation result
                validation_result = {
                    "validation_status": validation_status,
                    "extraction_result": result,
                    "database_matches": closest_matches,
                    "closest_matches": [
                        {
                            "db_serial": match[0],
                            "db_reading": match[1],
                            "db_date": match[2],
                            "db_unit": match[3],
                            "percentage_error": match[4],
                            "match_type": match[5] if len(match) > 5 else "unknown"
                        }
                        for match in closest_matches[:3]  # Top 3 matches
                    ]
                }
                
                validation_results.append(validation_result)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Database validation error: {e}")
            for result in extraction_results:
                validation_results.append({
                    "validation_status": "error",
                    "error": str(e),
                    "extraction_result": result
                })
        
        return validation_results
    
    def _enhanced_fuzzy_matching(self, cursor, reading: str, extracted_serial: str, extracted_date: str, tolerance_percentage: float) -> List[tuple]:
        """
        Enhanced fuzzy matching with multiple strategies
        
        Args:
            cursor: Database cursor
            reading: Extracted meter reading
            extracted_serial: Extracted meter serial
            extracted_date: Extracted date
            tolerance_percentage: Tolerance for percentage matching
            
        Returns:
            List of matches with scores
        """
        matches = []
        
        if not reading:
            return matches
        
        try:
            # Strategy 1: Exact serial match
            if extracted_serial:
                cursor.execute("""
                    SELECT meter_serial_number, meter_reading, reading_date, reading_unit
                    FROM meter_readings 
                    WHERE meter_serial_number = ?
                """, (extracted_serial,))
                
                for row in cursor.fetchall():
                    db_reading = str(row[1])
                    percentage_error = self._calculate_percentage_error(reading, db_reading)
                    if percentage_error <= tolerance_percentage:
                        matches.append((row[0], row[1], row[2], row[3], percentage_error, "exact_serial"))
            
            # Strategy 2: Partial serial match
            if extracted_serial and len(extracted_serial) >= 6:
                partial_serial = extracted_serial[:6]
                cursor.execute("""
                    SELECT meter_serial_number, meter_reading, reading_date, reading_unit
                    FROM meter_readings 
                    WHERE meter_serial_number LIKE ?
                """, (f"{partial_serial}%",))
                
                for row in cursor.fetchall():
                    db_reading = str(row[1])
                    percentage_error = self._calculate_percentage_error(reading, db_reading)
                    if percentage_error <= tolerance_percentage:
                        matches.append((row[0], row[1], row[2], row[3], percentage_error, "partial_serial"))
            
            # Strategy 3: Reading-based match with date proximity
            if extracted_date:
                cursor.execute("""
                    SELECT meter_serial_number, meter_reading, reading_date, reading_unit
                    FROM meter_readings 
                    WHERE reading_date = ?
                """, (extracted_date,))
                
                for row in cursor.fetchall():
                    db_reading = str(row[1])
                    percentage_error = self._calculate_percentage_error(reading, db_reading)
                    if percentage_error <= tolerance_percentage:
                        matches.append((row[0], row[1], row[2], row[3], percentage_error, "date_match"))
            
            # Strategy 4: General reading match
            cursor.execute("""
                SELECT meter_serial_number, meter_reading, reading_date, reading_unit
                FROM meter_readings 
                WHERE ABS(CAST(meter_reading AS INTEGER) - ?) <= ?
            """, (int(reading), int(reading) * tolerance_percentage / 100))
            
            for row in cursor.fetchall():
                db_reading = str(row[1])
                percentage_error = self._calculate_percentage_error(reading, db_reading)
                if percentage_error <= tolerance_percentage:
                    matches.append((row[0], row[1], row[2], row[3], percentage_error, "tolerance"))
            
            # Sort by percentage error and remove duplicates
            unique_matches = {}
            for match in matches:
                key = (match[0], match[1])  # serial, reading
                if key not in unique_matches or match[4] < unique_matches[key][4]:
                    unique_matches[key] = match
            
            return sorted(unique_matches.values(), key=lambda x: x[4])[:10]  # Top 10 matches
            
        except Exception as e:
            logger.error(f"Fuzzy matching error: {e}")
            return matches
    
    def _calculate_percentage_error(self, reading1: str, reading2: str) -> float:
        """
        Calculate percentage error between two readings
        
        Args:
            reading1: First reading
            reading2: Second reading
            
        Returns:
            Percentage error
        """
        try:
            val1 = float(reading1)
            val2 = float(reading2)
            if val2 == 0:
                return 100.0
            return abs(val1 - val2) / val2 * 100
        except:
            return 100.0
    
    def generate_enhanced_accuracy_report(self, validation_results: List[Dict]) -> Dict:
        """
        Generate comprehensive accuracy report
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Accuracy report dictionary
        """
        total_processed = len(validation_results)
        successful_matches = 0
        exact_matches = 0
        close_matches = 0
        fuzzy_matches = 0
        failed_matches = 0
        errors = 0
        
        total_confidence = 0
        total_datetime_confidence = 0
        datetime_extractions = 0
        
        for result in validation_results:
            if "error" in result:
                errors += 1
                continue
            
            status = result.get("validation_status", "no_match")
            if status == "exact_match":
                successful_matches += 1
                exact_matches += 1
            elif status == "close_match":
                successful_matches += 1
                close_matches += 1
            elif status == "fuzzy_match":
                successful_matches += 1
                fuzzy_matches += 1
            else:
                failed_matches += 1
            
            # Collect confidence scores
            extraction = result.get("extraction_result", {})
            confidence = extraction.get("confidence", 0)
            total_confidence += confidence
            
            datetime_confidence = extraction.get("date_time_confidence", 0)
            if datetime_confidence > 0:
                total_datetime_confidence += datetime_confidence
                datetime_extractions += 1
        
        avg_confidence = total_confidence / total_processed if total_processed > 0 else 0
        avg_datetime_confidence = total_datetime_confidence / datetime_extractions if datetime_extractions > 0 else 0
        
        return {
            "processing_summary": {
                "total_processed": total_processed,
                "successful_matches": successful_matches,
                "exact_matches": exact_matches,
                "close_matches": close_matches,
                "fuzzy_matches": fuzzy_matches,
                "failed_matches": failed_matches,
                "errors": errors,
                "success_rate": (successful_matches / total_processed) * 100 if total_processed > 0 else 0
            },
            "accuracy_metrics": {
                "average_confidence": round(avg_confidence, 2),
                "average_datetime_confidence": round(avg_datetime_confidence, 2),
                "datetime_extractions": datetime_extractions,
                "datetime_extraction_rate": (datetime_extractions / total_processed) * 100 if total_processed > 0 else 0
            },
            "recommendations": self._generate_enhanced_recommendations(
                successful_matches, total_processed, datetime_extractions, 
                avg_confidence, avg_datetime_confidence
            )
        }
    
    def _generate_enhanced_recommendations(self, successful_matches: int, total_processed: int, 
                                         datetime_extractions: int, avg_confidence: float, 
                                         avg_datetime_confidence: float) -> List[str]:
        """
        Generate recommendations based on processing results
        
        Args:
            successful_matches: Number of successful matches
            total_processed: Total number of processed images
            datetime_extractions: Number of successful datetime extractions
            avg_confidence: Average confidence score
            avg_datetime_confidence: Average datetime confidence score
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        success_rate = (successful_matches / total_processed) * 100 if total_processed > 0 else 0
        
        if success_rate < 50:
            recommendations.append("Low success rate detected. Consider improving image quality or adjusting extraction parameters.")
        
        if avg_confidence < 6:
            recommendations.append("Low average confidence scores. Review image quality and extraction settings.")
        
        if datetime_extractions < total_processed * 0.3:
            recommendations.append("Low datetime extraction rate. Consider enhancing temporal extraction capabilities.")
        
        if avg_datetime_confidence < 5:
            recommendations.append("Low datetime confidence scores. Review temporal extraction accuracy.")
        
        if not recommendations:
            recommendations.append("Processing results are within acceptable parameters.")
        
        return recommendations
    
    def save_results(self, data: Dict, filename: str = None) -> None:
        """
        Save results to file
        
        Args:
            data: Data to save
            filename: Output filename (optional)
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bedrock_ocr_results_{timestamp}.json"
        
        output_path = Path("bedrock_ocr/results") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")


def main():
    """
    Main function for testing the Claude Vision meter reader
    """
    # Example usage
    reader = ClaudeVisionMeterReader()
    
    # Test single image processing
    test_image = "path/to/test/image.jpg"
    if os.path.exists(test_image):
        result = reader.extract_meter_data(test_image)
        print("Single image result:", json.dumps(result, indent=2))
    
    # Test batch processing
    test_images = ["path/to/image1.jpg", "path/to/image2.jpg"]
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if existing_images:
        # Synchronous batch processing
        batch_results = reader.process_batch_sync(existing_images)
        print(f"Batch processing completed: {len(batch_results)} results")
        
        # Validate results
        validation_results = reader.validate_against_database(batch_results)
        
        # Generate report
        accuracy_report = reader.generate_enhanced_accuracy_report(validation_results)
        print("Accuracy report:", json.dumps(accuracy_report, indent=2))
        
        # Save results
        reader.save_results({
            "batch_results": batch_results,
            "validation_results": validation_results,
            "accuracy_report": accuracy_report
        })


if __name__ == "__main__":
    main() 