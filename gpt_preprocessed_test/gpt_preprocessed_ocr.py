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
import sys
import shutil

# Add parent directory to path to import preprocessing module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from img_preprocessing import MeterImagePreprocessor

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT4VisionPreprocessedMeterReader:
    """
    Enhanced Smart Meter Reading using GPT-4 Vision API with PREPROCESSED images
    Integrates with existing SQLite database for validation
    Uses preprocessed images from img_preprocessing.py for better results
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
        self.max_tokens = int(os.getenv("MAX_TOKENS", "800"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.detail_level = os.getenv("DETAIL_LEVEL", "high")
        
        # Initialize preprocessor
        self.preprocessor = MeterImagePreprocessor()
        
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
        
        logger.info(f"Initialized GPT-4 Vision Preprocessed Reader with model: {self.model_name}")
        logger.info(f"Database path: {self.db_path}")
        logger.info(f"Using preprocessed images for enhanced accuracy")
    
    def preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image using the same pipeline as img_preprocessing.py
        Returns path to the preprocessed image
        """
        try:
            logger.info(f"Preprocessing image: {image_path}")
            
            # Use the preprocessor to create preprocessed image
            binary_image, gray_image = self.preprocessor.preprocess_image(image_path, save_intermediate=True)
            
            # The preprocessor saves the processed image as {original_name}_processed.png
            base_name = Path(image_path).stem
            processed_path = f"{base_name}_processed.png"
            
            logger.info(f"Preprocessed image saved as: {processed_path}")
            return processed_path
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            # Fallback to original image if preprocessing fails
            return image_path
    
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
        Optimized for preprocessed images
        """
        return """
        You are an expert in reading digital utility meters from PREPROCESSED images. 
        This image has been enhanced for better text recognition with the following improvements:
        - Resolution enhancement
        - Noise removal
        - Contrast enhancement
        - Adaptive thresholding
        - Deskewing correction
        
        Analyze this preprocessed meter image and extract the following information:

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
        12. PREPROCESSING_EFFECTIVENESS: Rate how well the preprocessing helped (1-10)

        SPECIAL INSTRUCTIONS FOR PREPROCESSED IMAGES:
        - The image has been enhanced for better text recognition
        - Look for clear, sharp digits that should be easier to read
        - Pay attention to any artifacts or distortions from preprocessing
        - The contrast should be improved, making digits more prominent
        - Date/time information should be more readable if present

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
            "preprocessing_effectiveness": rating_of_preprocessing_help,
            "extraction_notes": "any_observations_or_uncertainties",
            "temporal_analysis": "description of all date/time elements found"
        }

        Focus on accuracy and be explicit about any uncertainty in your reading or date/time extraction.
        If no date/time information is visible, explicitly state this in the response.
        """
    
    def extract_meter_data(self, image_path: str, detail_level: str = None) -> Dict:
        """
        Extract meter reading and date/time using GPT-4 Vision with PREPROCESSED image
        
        Args:
            image_path: Path to original meter image
            detail_level: "low", "high", or "auto" (overrides environment default)
            
        Returns:
            Dictionary containing extracted meter data including date/time
        """
        try:
            # Step 1: Preprocess the image
            preprocessed_path = self.preprocess_image(image_path)
            
            # Step 2: Use provided detail level or environment default
            detail = detail_level or self.detail_level
            
            # Step 3: Encode preprocessed image
            base64_image = self.encode_image(preprocessed_path)
            
            # Step 4: Create API request
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
            
            # Step 5: Parse response
            response_text = response.choices[0].message.content
            
            # Step 6: Try to parse JSON response
            try:
                meter_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: extract reading and date/time from text response
                logger.warning("Failed to parse JSON response, attempting text extraction")
                meter_data = self._enhanced_fallback_extraction(response_text)
            
            # Step 7: Add preprocessing metadata
            meter_data.update({
                'original_image_path': image_path,
                'preprocessed_image_path': preprocessed_path,
                'processing_timestamp': datetime.now().isoformat(),
                'preprocessing_used': True
            })
            
            # Step 8: Validate date/time data
            meter_data = self._validate_datetime_data(meter_data)
            
            return meter_data
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'error': str(e),
                'original_image_path': image_path,
                'processing_timestamp': datetime.now().isoformat(),
                'preprocessing_used': True
            }
    
    def _enhanced_fallback_extraction(self, response_text: str) -> Dict:
        """Enhanced fallback extraction for non-JSON responses"""
        meter_data = {
            'meter_reading': None,
            'meter_type': 'unknown',
            'meter_serial': None,
            'units': 'unknown',
            'display_type': 'unknown',
            'date': None,
            'time': None,
            'timestamp_format': None,
            'timestamp_source': None,
            'additional_timestamps': [],
            'additional_text': response_text,
            'confidence': 5,
            'date_time_confidence': 3,
            'preprocessing_effectiveness': 5,
            'extraction_notes': 'Fallback extraction used - JSON parsing failed',
            'temporal_analysis': 'Unable to extract temporal information from fallback'
        }
        
        # Extract readings using patterns
        for pattern in self.reading_patterns:
            matches = re.findall(pattern, response_text)
            if matches:
                meter_data['meter_reading'] = matches[0]
                break
        
        # Extract date/time using patterns
        for pattern in self.datetime_patterns:
            matches = re.findall(pattern, response_text)
            if matches:
                if ':' in matches[0]:  # Time format
                    meter_data['time'] = matches[0]
                else:  # Date format
                    meter_data['date'] = matches[0]
        
        return meter_data
    
    def _validate_datetime_data(self, meter_data: Dict) -> Dict:
        """Validate and enhance date/time data"""
        # Add validation metadata
        meter_data['datetime_validation'] = {
            'has_date': bool(meter_data.get('date')),
            'has_time': bool(meter_data.get('time')),
            'has_timestamp': bool(meter_data.get('date') or meter_data.get('time')),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        return meter_data
    
    def process_directory(self, images_dir: str = None) -> List[Dict]:
        """Process all images in directory with preprocessing"""
        results = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Use provided directory or default to parent images directory
        if not images_dir:
            images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
        
        images_path = Path(images_dir)
        if not images_path.exists():
            logger.error(f"Directory not found: {images_dir}")
            return results
        
        image_files = [f for f in images_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        logger.info(f"Found {len(image_files)} images to process with preprocessing")
        
        for image_file in image_files:
            logger.info(f"Processing with preprocessing: {image_file.name}")
            result = self.extract_meter_data(str(image_file))
            if result:
                results.append(result)
        
        # Clean up preprocessor temporary files
        self.preprocessor.cleanup()
        
        return results
    
    def validate_against_database(self, extraction_results: List[Dict]) -> List[Dict]:
        """Validate OCR results against database records"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            validation_results = []
            
            for result in extraction_results:
                if result.get('meter_reading'):
                    reading = result['meter_reading']
                    
                    # Try to find matching record in database
                    cursor.execute("""
                        SELECT meter_serial_number, meter_reading, reading_date 
                        FROM meter_readings 
                        WHERE meter_reading = ? OR original_reading = ?
                    """, (reading, reading))
                    
                    matches = cursor.fetchall()
                    
                    validation_results.append({
                        'original_image_path': result.get('original_image_path'),
                        'preprocessed_image_path': result.get('preprocessed_image_path'),
                        'gpt4_reading': reading,
                        'gpt4_confidence': result.get('confidence', 0),
                        'gpt4_date': result.get('date'),
                        'gpt4_time': result.get('time'),
                        'gpt4_datetime_confidence': result.get('date_time_confidence', 0),
                        'preprocessing_effectiveness': result.get('preprocessing_effectiveness', 0),
                        'database_matches': matches,
                        'validation_status': 'match' if matches else 'no_match',
                        'processing_timestamp': result.get('processing_timestamp')
                    })
            
            conn.close()
            return validation_results
            
        except Exception as e:
            logger.error(f"Database validation error: {e}")
            return []
    
    def save_results(self, data: Dict, filename: str = None) -> None:
        """Save results to JSON file"""
        try:
            # Create output directory
            output_dir = "gpt_preprocessed_test/output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"gpt4_preprocessed_results_{timestamp}.json"
            
            filepath = Path(output_dir) / filename
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    """
    Main function demonstrating GPT-4 Vision meter reading with PREPROCESSED images
    """
    try:
        # Initialize the preprocessed meter reader
        meter_reader = GPT4VisionPreprocessedMeterReader()
        
        print("=== GPT-4 Vision with Preprocessed Images ===")
        print("Processing images with preprocessing pipeline...")
        
        # Process all images in the images directory
        batch_results = meter_reader.process_directory()
        
        # Validate against database
        print("Validating against database...")
        validation_results = meter_reader.validate_against_database(batch_results)
        
        # Save results
        meter_reader.save_results(batch_results, "gpt4_preprocessed_extractions.json")
        meter_reader.save_results(validation_results, "gpt4_preprocessed_validation.json")
        
        # Print summary
        print(f"\n=== Processing Summary ===")
        print(f"Images processed: {len(batch_results)}")
        print(f"Successful extractions: {len([r for r in batch_results if r.get('meter_reading')])}")
        print(f"Database matches: {len([r for r in validation_results if r['validation_status'] == 'match'])}")
        
        if batch_results:
            avg_confidence = np.mean([r.get('confidence', 0) for r in batch_results])
            avg_preprocessing_effectiveness = np.mean([r.get('preprocessing_effectiveness', 0) for r in batch_results])
            print(f"Average confidence: {avg_confidence:.2f}")
            print(f"Average preprocessing effectiveness: {avg_preprocessing_effectiveness:.2f}")
        
        print(f"\nResults saved in: gpt_preprocessed_test/output/")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 