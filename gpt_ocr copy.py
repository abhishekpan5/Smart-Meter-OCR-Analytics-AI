import os
import base64
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from openai import OpenAI
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT4VisionMeterReader:
    """
    Enhanced Smart Meter Reading using GPT-4 Vision API with date/time extraction
    Integrates with existing SQLite database for validation
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize GPT-4 Vision client with environment variables
        
        Args:
            db_path: Path to SQLite database (defaults to environment variable)
        """
        # Load API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
        
        self.client = OpenAI(api_key=api_key)
        
        # Load database path from environment or use default
        self.db_path = db_path or os.getenv("DATABASE_PATH", "smart_meter_database.db")
        
        # Load configuration from environment variables
        self.model_name = os.getenv("GPT_MODEL", "gpt-4o")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "800"))  # Increased for date/time extraction
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.detail_level = os.getenv("DETAIL_LEVEL", "high")
        
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
        
        logger.info(f"Initialized GPT-4 Vision Reader with model: {self.model_name}")
        logger.info(f"Database path: {self.db_path}")
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string for API
        
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
        Create a detailed prompt for meter reading and date/time extraction
        
        Returns:
            Enhanced structured prompt for GPT-4 Vision
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
        10. CONFIDENCE: Your confidence level (1-10) in the reading accuracy
        11. DATE_TIME_CONFIDENCE: Your confidence level (1-10) specifically for date/time extraction

        SPECIAL INSTRUCTIONS FOR DATE/TIME EXTRACTION:
        - Look carefully for any digital display showing date/time information
        - Common locations: top/bottom of LCD display, separate date/time panel
        - Pay attention to timestamp watermarks or overlay text
        - Note if date/time appears to be from the meter itself or from a camera/phone
        - If multiple timestamps are visible, extract all of them with their sources

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
            "confidence": confidence_score_for_reading,
            "date_time_confidence": confidence_score_for_datetime,
            "extraction_notes": "any_observations_or_uncertainties",
            "temporal_analysis": "description of all date/time elements found"
        }

        Focus on accuracy and be explicit about any uncertainty in your reading or date/time extraction.
        If no date/time information is visible, explicitly state this in the response.
        """
    
    def extract_meter_data(self, image_path: str, detail_level: str = None) -> Dict:
        """
        Extract meter reading and date/time using GPT-4 Vision
        
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
            
            # Create API request with environment-configured parameters
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.create_enhanced_prompt()
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": detail
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
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
            
            # Add metadata
            meter_data.update({
                "image_path": image_path,
                "extraction_timestamp": datetime.now().isoformat(),
                "api_model": self.model_name,
                "processing_tokens": response.usage.total_tokens,
                "raw_response": response_text,
                "configuration": {
                    "detail_level": detail,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
            })
            
            return meter_data
            
        except Exception as e:
            logger.error(f"Error extracting meter data from {image_path}: {e}")
            return {
                "error": str(e),
                "image_path": image_path,
                "extraction_timestamp": datetime.now().isoformat()
            }
    
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
            "confidence": 5,  # Medium confidence for fallback
            "date_time_confidence": 3,  # Lower confidence for fallback date/time
            "extraction_notes": "Fallback extraction used due to JSON parsing failure",
            "raw_response": response_text
        }
    
    def _validate_datetime_data(self, meter_data: Dict) -> Dict:
        """
        Validate and normalize extracted date/time data
        
        Args:
            meter_data: Dictionary containing extracted meter data
            
        Returns:
            Dictionary with validated and normalized date/time data
        """
        try:
            # Attempt to parse and validate date
            if meter_data.get('date'):
                date_str = meter_data['date']
                parsed_date = None
                
                # Try different date formats
                date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        meter_data['parsed_date'] = parsed_date.isoformat()
                        meter_data['date_format_used'] = fmt
                        break
                    except ValueError:
                        continue
                
                if not parsed_date:
                    meter_data['date_parsing_error'] = f"Could not parse date: {date_str}"
            
            # Attempt to parse and validate time
            if meter_data.get('time'):
                time_str = meter_data['time']
                parsed_time = None
                
                # Try different time formats
                time_formats = ['%H:%M:%S', '%H:%M', '%I:%M %p', '%I:%M:%S %p']
                for fmt in time_formats:
                    try:
                        parsed_time = datetime.strptime(time_str, fmt).time()
                        meter_data['parsed_time'] = parsed_time.isoformat()
                        meter_data['time_format_used'] = fmt
                        break
                    except ValueError:
                        continue
                
                if not parsed_time:
                    meter_data['time_parsing_error'] = f"Could not parse time: {time_str}"
            
            # Create combined datetime if both date and time are available
            if meter_data.get('parsed_date') and meter_data.get('parsed_time'):
                try:
                    combined_dt = f"{meter_data['parsed_date'].split('T')[0]}T{meter_data['parsed_time']}"
                    meter_data['combined_datetime'] = combined_dt
                except Exception as e:
                    meter_data['datetime_combination_error'] = str(e)
            
        except Exception as e:
            meter_data['datetime_validation_error'] = str(e)
        
        return meter_data
    
    def process_directory(self, images_dir: str = None) -> List[Dict]:
        """
        Process all meter images in a directory with enhanced date/time extraction
        
        Args:
            images_dir: Directory containing meter images (defaults to environment variable)
            
        Returns:
            List of extraction results including date/time data
        """
        # Use provided directory or environment default
        images_directory = images_dir or os.getenv("IMAGES_DIRECTORY", "images")
        
        results = []
        image_extensions = set(os.getenv("IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.bmp,.tiff,.webp").split(","))
        
        images_path = Path(images_directory)
        if not images_path.exists():
            logger.error(f"Directory not found: {images_directory}")
            return results
        
        image_files = [f for f in images_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        logger.info(f"Found {len(image_files)} images to process in {images_directory}")
        
        for image_file in image_files:
            logger.info(f"Processing: {image_file.name}")
            result = self.extract_meter_data(str(image_file))
            if result:
                results.append(result)
        
        return results
    
    def validate_against_database(self, extraction_results: List[Dict]) -> List[Dict]:
        """
        Enhanced validation including date/time comparison against database records
        
        Args:
            extraction_results: List of extraction results
            
        Returns:
            List of validation results including temporal validation
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load tolerance from environment
            tolerance_percentage = float(os.getenv("VALIDATION_TOLERANCE", "0.1"))  # 10% default
            
            validation_results = []
            
            for result in extraction_results:
                if result.get('error'):
                    continue
                    
                reading = result.get('meter_reading')
                extracted_date = result.get('date')
                extracted_time = result.get('time')
                
                if not reading:
                    continue
                
                # Normalize reading by removing leading zeros and converting to numeric
                try:
                    # Remove leading zeros and convert to numeric
                    normalized_reading = str(reading).lstrip('0')
                    if not normalized_reading:  # If all zeros, keep as 0
                        normalized_reading = '0'
                    numeric_reading = float(normalized_reading.replace(',', ''))
                except (ValueError, TypeError):
                    continue
                
                # Try multiple comparison strategies
                matches = []
                
                # Strategy 1: Exact string match (handles leading zeros)
                cursor.execute("""
                    SELECT meter_serial_number, meter_reading, reading_unit, 
                           reading_date, 0 as difference
                    FROM meter_readings 
                    WHERE meter_reading = ?
                    ORDER BY reading_date DESC 
                    LIMIT 5
                """, (str(reading),))
                
                exact_matches = cursor.fetchall()
                matches.extend(exact_matches)
                
                # Strategy 2: Try with leading zero if reading doesn't have one
                if not str(reading).startswith('0'):
                    cursor.execute("""
                        SELECT meter_serial_number, meter_reading, reading_unit, 
                               reading_date, 0 as difference
                        FROM meter_readings 
                        WHERE meter_reading = ?
                        ORDER BY reading_date DESC 
                        LIMIT 5
                    """, ('0' + str(reading),))
                    
                    leading_zero_matches = cursor.fetchall()
                    matches.extend(leading_zero_matches)
                
                # Strategy 3: Try without leading zero if reading has one
                if str(reading).startswith('0'):
                    cursor.execute("""
                        SELECT meter_serial_number, meter_reading, reading_unit, 
                               reading_date, 0 as difference
                        FROM meter_readings 
                        WHERE meter_reading = ?
                        ORDER BY reading_date DESC 
                        LIMIT 5
                    """, (str(reading).lstrip('0'),))
                    
                    no_leading_zero_matches = cursor.fetchall()
                    matches.extend(no_leading_zero_matches)
                
                # Strategy 4: Numeric match (convert to numeric for comparison)
                try:
                    numeric_reading = float(str(reading).replace(',', ''))
                    cursor.execute("""
                        SELECT meter_serial_number, meter_reading, reading_unit, 
                               reading_date, ABS(CAST(meter_reading AS REAL) - ?) as difference
                        FROM meter_readings 
                        WHERE CAST(meter_reading AS REAL) = ?
                        ORDER BY reading_date DESC 
                        LIMIT 5
                    """, (numeric_reading, numeric_reading))
                    
                    numeric_matches = cursor.fetchall()
                    matches.extend(numeric_matches)
                except (ValueError, TypeError):
                    pass
                
                # Strategy 5: Tolerance-based match (for close readings)
                try:
                    numeric_reading = float(str(reading).replace(',', ''))
                    cursor.execute("""
                        SELECT meter_serial_number, meter_reading, reading_unit, 
                               reading_date, ABS(CAST(meter_reading AS REAL) - ?) as difference
                        FROM meter_readings 
                        WHERE ABS(CAST(meter_reading AS REAL) - ?) <= ?
                        ORDER BY difference ASC 
                        LIMIT 5
                    """, (numeric_reading, numeric_reading, numeric_reading * tolerance_percentage))
                    
                    tolerance_matches = cursor.fetchall()
                    matches.extend(tolerance_matches)
                except (ValueError, TypeError):
                    pass
                
                # Remove duplicates while preserving order
                seen = set()
                unique_matches = []
                for match in matches:
                    match_key = (match[0], match[1], match[2], match[3])  # serial, reading, unit, date
                    if match_key not in seen:
                        seen.add(match_key)
                        unique_matches.append(match)
                
                # Enhanced validation result with date/time analysis
                validation_result = {
                    "image_path": result.get("image_path"),
                    "gpt4_reading": reading,
                    "gpt4_confidence": result.get("confidence", 0),
                    "gpt4_date": extracted_date,
                    "gpt4_time": extracted_time,
                    "gpt4_datetime_confidence": result.get("date_time_confidence", 0),
                    "database_matches": len(unique_matches),
                    "closest_matches": [
                        {
                            "db_serial": match[0],
                            "db_reading": match[1],
                            "db_unit": match[2],
                            "db_date": match[3],
                            "difference": match[4],
                            "percentage_error": (match[4] / float(match[1])) * 100 if float(match[1]) > 0 else 0,
                            "date_comparison": self._compare_dates(extracted_date, match[3])
                        }
                        for match in unique_matches
                    ],
                    "validation_status": "match" if unique_matches else "no_match",
                    "temporal_validation": self._validate_temporal_consistency(result, unique_matches),
                    "extraction_data": result
                }
                
                validation_results.append(validation_result)
            
            conn.close()
            return validation_results
            
        except Exception as e:
            logger.error(f"Database validation error: {e}")
            return []
    
    def _compare_dates(self, extracted_date: str, db_date: str) -> Dict:
        """
        Compare extracted date with database date
        
        Args:
            extracted_date: Date extracted from image
            db_date: Date from database
            
        Returns:
            Dictionary with date comparison results
        """
        if not extracted_date or not db_date:
            return {"status": "missing_data", "comparison": None}
        
        try:
            # Parse dates for comparison
            extracted_dt = datetime.fromisoformat(extracted_date.replace('Z', '+00:00'))
            db_dt = datetime.fromisoformat(db_date.replace('Z', '+00:00'))
            
            difference = abs((extracted_dt - db_dt).days)
            
            return {
                "status": "compared",
                "days_difference": difference,
                "match_level": "exact" if difference == 0 else "close" if difference <= 1 else "different"
            }
        except Exception as e:
            return {"status": "parsing_error", "error": str(e)}
    
    def _validate_temporal_consistency(self, result: Dict, db_matches: List) -> Dict:
        """
        Validate temporal consistency between extracted and database data
        
        Args:
            result: Extraction result with date/time data
            db_matches: List of database matches
            
        Returns:
            Dictionary with temporal validation results
        """
        validation = {
            "has_extracted_datetime": bool(result.get('date') or result.get('time')),
            "datetime_source": result.get('timestamp_source', 'unknown'),
            "consistency_score": 0,
            "notes": []
        }
        
        if result.get('combined_datetime') and db_matches:
            # Compare with closest database match
            closest_match = db_matches[0]
            db_date = closest_match[3]  # reading_date from database
            
            date_comparison = self._compare_dates(result.get('combined_datetime'), db_date)
            validation['date_comparison'] = date_comparison
            
            if date_comparison.get('match_level') == 'exact':
                validation['consistency_score'] = 10
                validation['notes'].append("Extracted date/time matches database exactly")
            elif date_comparison.get('match_level') == 'close':
                validation['consistency_score'] = 8
                validation['notes'].append("Extracted date/time is close to database date")
            else:
                validation['consistency_score'] = 5
                validation['notes'].append("Extracted date/time differs significantly from database")
        
        return validation
    
    def generate_enhanced_accuracy_report(self, validation_results: List[Dict]) -> Dict:
        """
        Generate comprehensive accuracy report including date/time extraction metrics
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary containing enhanced accuracy metrics
        """
        if not validation_results:
            return {"error": "No validation results to analyze"}
        
        total_processed = len(validation_results)
        successful_matches = sum(1 for r in validation_results if r["validation_status"] == "match")
        
        # Date/time extraction statistics
        datetime_extractions = sum(1 for r in validation_results 
                                 if r.get('gpt4_date') or r.get('gpt4_time'))
        successful_datetime = sum(1 for r in validation_results 
                                if r.get('gpt4_datetime_confidence', 0) >= 5)
        
        # Load accuracy thresholds from environment
        close_match_threshold = float(os.getenv("CLOSE_MATCH_THRESHOLD", "1.0"))  # 1% default
        
        # Calculate accuracy metrics
        exact_matches = 0
        close_matches = 0
        confidence_scores = []
        datetime_confidence_scores = []
        percentage_errors = []
        temporal_consistency_scores = []
        
        for result in validation_results:
            confidence_scores.append(result.get("gpt4_confidence", 0))
            datetime_confidence_scores.append(result.get("gpt4_datetime_confidence", 0))
            
            # Temporal validation scoring
            if result.get('temporal_validation'):
                temporal_consistency_scores.append(result['temporal_validation'].get('consistency_score', 0))
            
            if result["closest_matches"]:
                best_match = result["closest_matches"][0]
                percentage_error = best_match["percentage_error"]
                percentage_errors.append(percentage_error)
                
                if percentage_error == 0:
                    exact_matches += 1
                elif percentage_error <= close_match_threshold:
                    close_matches += 1
        
        # Compile enhanced report
        report = {
            "processing_summary": {
                "total_images": total_processed,
                "successful_extractions": successful_matches,
                "extraction_rate": (successful_matches / total_processed) * 100 if total_processed > 0 else 0,
                "datetime_extractions": datetime_extractions,
                "datetime_extraction_rate": (datetime_extractions / total_processed) * 100 if total_processed > 0 else 0,
                "successful_datetime": successful_datetime
            },
            "accuracy_metrics": {
                "exact_matches": exact_matches,
                "close_matches": close_matches,
                "exact_accuracy": (exact_matches / successful_matches) * 100 if successful_matches > 0 else 0,
                "close_accuracy": ((exact_matches + close_matches) / successful_matches) * 100 if successful_matches > 0 else 0,
                "average_confidence": np.mean(confidence_scores) if confidence_scores else 0,
                "average_datetime_confidence": np.mean(datetime_confidence_scores) if datetime_confidence_scores else 0
            },
            "temporal_analysis": {
                "datetime_extraction_success_rate": (successful_datetime / total_processed) * 100 if total_processed > 0 else 0,
                "average_temporal_consistency": np.mean(temporal_consistency_scores) if temporal_consistency_scores else 0,
                "temporal_validation_available": len(temporal_consistency_scores)
            },
            "error_analysis": {
                "mean_percentage_error": np.mean(percentage_errors) if percentage_errors else 0,
                "max_percentage_error": max(percentage_errors) if percentage_errors else 0,
                "std_percentage_error": np.std(percentage_errors) if percentage_errors else 0
            },
            "configuration": {
                "model": self.model_name,
                "close_match_threshold": close_match_threshold,
                "validation_tolerance": os.getenv("VALIDATION_TOLERANCE", "0.1"),
                "enhanced_datetime_extraction": True
            },
            "recommendations": self._generate_enhanced_recommendations(
                successful_matches, total_processed, datetime_extractions,
                np.mean(confidence_scores) if confidence_scores else 0,
                np.mean(datetime_confidence_scores) if datetime_confidence_scores else 0
            )
        }
        
        return report
    
    def _generate_enhanced_recommendations(self, successful_matches: int, total_processed: int, 
                                         datetime_extractions: int, avg_confidence: float, 
                                         avg_datetime_confidence: float) -> List[str]:
        """Generate enhanced recommendations including date/time extraction guidance"""
        recommendations = []
        
        success_rate = (successful_matches / total_processed) * 100 if total_processed > 0 else 0
        datetime_rate = (datetime_extractions / total_processed) * 100 if total_processed > 0 else 0
        
        # Load recommendation thresholds from environment
        min_success_rate = float(os.getenv("MIN_SUCCESS_RATE", "90"))
        min_confidence = float(os.getenv("MIN_CONFIDENCE", "7"))
        min_datetime_confidence = float(os.getenv("MIN_DATETIME_CONFIDENCE", "6"))
        
        if success_rate < min_success_rate:
            recommendations.append("Consider implementing image preprocessing to improve image quality")
            recommendations.append("Use 'high' detail level for better accuracy on complex meter displays")
        
        if avg_confidence < min_confidence:
            recommendations.append("Review images with low confidence scores manually")
            recommendations.append("Consider combining GPT-4 Vision with traditional OCR for consensus")
        
        if datetime_rate < 50:
            recommendations.append("Date/time extraction rate is low - ensure images include timestamp displays")
            recommendations.append("Consider capturing images that show the meter's built-in date/time display")
        
        if avg_datetime_confidence < min_datetime_confidence:
            recommendations.append("Date/time extraction confidence is low - review timestamp visibility in images")
            recommendations.append("Ensure adequate lighting and focus on date/time display areas")
        
        if success_rate > 95 and datetime_rate > 80:
            recommendations.append("Excellent performance in both reading and date/time extraction!")
        
        return recommendations
    
    def save_results(self, data: Dict, filename: str = None) -> None:
        """Save results to JSON file with configurable output directory"""
        try:
            # Use environment variable for output directory
            output_dir = os.getenv("OUTPUT_DIRECTORY", "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"gpt4_vision_enhanced_results_{timestamp}.json"
            
            filepath = Path(output_dir) / filename
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    """
    Main function demonstrating enhanced GPT-4 Vision meter reading with date/time extraction
    """
    try:
        # Initialize the enhanced meter reader
        meter_reader = GPT4VisionMeterReader()
        
        # Get processing mode from environment
        processing_mode = os.getenv("PROCESSING_MODE", "batch")
        
        if processing_mode == "single":
            # Process single image
            single_image_path = os.getenv("SINGLE_IMAGE_PATH", "images/sample_meter.jpg")
            print(f"Processing single image: {single_image_path}")
            single_result = meter_reader.extract_meter_data(single_image_path)
            print(f"Single image result: {json.dumps(single_result, indent=2)}")
            meter_reader.save_results([single_result], "single_image_enhanced_result.json")
            
        elif processing_mode == "batch":
            # Process entire directory
            print("Processing directory of images with enhanced date/time extraction...")
            batch_results = meter_reader.process_directory()
            
            # Validate against database if enabled
            if os.getenv("ENABLE_VALIDATION", "true").lower() == "true":
                print("Validating against database with temporal analysis...")
                validation_results = meter_reader.validate_against_database(batch_results)
                
                # Generate enhanced accuracy report
                print("Generating enhanced accuracy report...")
                accuracy_report = meter_reader.generate_enhanced_accuracy_report(validation_results)
                
                # Save all results
                meter_reader.save_results(batch_results, "gpt4_vision_enhanced_extractions.json")
                meter_reader.save_results(validation_results, "gpt4_vision_enhanced_validation.json")
                meter_reader.save_results(accuracy_report, "gpt4_vision_enhanced_accuracy_report.json")
                
                # Print enhanced summary
                print(f"\n=== Enhanced GPT-4 Vision Processing Summary ===")
                print(f"Images processed: {len(batch_results)}")
                print(f"Successful extractions: {accuracy_report.get('processing_summary', {}).get('successful_extractions', 0)}")
                print(f"Extraction rate: {accuracy_report.get('processing_summary', {}).get('extraction_rate', 0):.2f}%")
                print(f"Date/time extractions: {accuracy_report.get('processing_summary', {}).get('datetime_extractions', 0)}")
                print(f"Date/time extraction rate: {accuracy_report.get('processing_summary', {}).get('datetime_extraction_rate', 0):.2f}%")
                print(f"Exact accuracy: {accuracy_report.get('accuracy_metrics', {}).get('exact_accuracy', 0):.2f}%")
                print(f"Average confidence: {accuracy_report.get('accuracy_metrics', {}).get('average_confidence', 0):.2f}")
                print(f"Average date/time confidence: {accuracy_report.get('accuracy_metrics', {}).get('average_datetime_confidence', 0):.2f}")
                print(f"Temporal consistency: {accuracy_report.get('temporal_analysis', {}).get('average_temporal_consistency', 0):.2f}")
            else:
                # Just save extraction results
                meter_reader.save_results(batch_results, "gpt4_vision_enhanced_extractions.json")
                print(f"Processed {len(batch_results)} images without validation")
        
        else:
            print(f"Unknown processing mode: {processing_mode}")
            print("Available modes: single, batch")
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
