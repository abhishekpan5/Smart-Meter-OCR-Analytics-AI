"""
Configuration file for Bedrock OCR System
"""

import os
from typing import Dict, Any

# AWS Configuration
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")

# Database Configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "smart_meter_database.db")

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Upload Configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "bedrock_uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10")) * 1024 * 1024  # 10MB
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))

# Processing Configuration
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# Validation Configuration
SERIAL_SIMILARITY_THRESHOLD = float(os.getenv("SERIAL_SIMILARITY_THRESHOLD", "0.7"))
READING_VALIDATION_RATIO_MIN = float(os.getenv("READING_VALIDATION_RATIO_MIN", "0.5"))
READING_VALIDATION_RATIO_MAX = float(os.getenv("READING_VALIDATION_RATIO_MAX", "1.5"))

# Confidence Configuration
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.5"))
PERFECT_MATCH_SERIAL_CONFIDENCE = float(os.getenv("PERFECT_MATCH_SERIAL_CONFIDENCE", "80.0"))
PERFECT_MATCH_READING_CONFIDENCE = float(os.getenv("PERFECT_MATCH_READING_CONFIDENCE", "80.0"))
GOOD_MATCH_READING_CONFIDENCE = float(os.getenv("GOOD_MATCH_READING_CONFIDENCE", "70.0"))
PARTIAL_MATCH_CONFIDENCE = float(os.getenv("PARTIAL_MATCH_CONFIDENCE", "50.0"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# CORS Configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# Image Quality Configuration
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "100.0"))
BRIGHTNESS_TARGET = int(os.getenv("BRIGHTNESS_TARGET", "128"))
CONTRAST_THRESHOLD = float(os.getenv("CONTRAST_THRESHOLD", "50.0"))

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "aws": {
            "region": AWS_REGION,
            "model": BEDROCK_MODEL
        },
        "database": {
            "path": DATABASE_PATH
        },
        "server": {
            "host": HOST,
            "port": PORT,
            "debug": DEBUG
        },
        "upload": {
            "directory": UPLOAD_DIR,
            "max_file_size": MAX_FILE_SIZE,
            "max_batch_size": MAX_BATCH_SIZE
        },
        "processing": {
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "request_timeout": REQUEST_TIMEOUT
        },
        "validation": {
            "serial_similarity_threshold": SERIAL_SIMILARITY_THRESHOLD,
            "reading_validation_ratio_min": READING_VALIDATION_RATIO_MIN,
            "reading_validation_ratio_max": READING_VALIDATION_RATIO_MAX
        },
        "confidence": {
            "min_threshold": MIN_CONFIDENCE_THRESHOLD,
            "perfect_match_serial": PERFECT_MATCH_SERIAL_CONFIDENCE,
            "perfect_match_reading": PERFECT_MATCH_READING_CONFIDENCE,
            "good_match_reading": GOOD_MATCH_READING_CONFIDENCE,
            "partial_match": PARTIAL_MATCH_CONFIDENCE
        },
        "logging": {
            "level": LOG_LEVEL,
            "format": LOG_FORMAT
        },
        "cors": {
            "allowed_origins": ALLOWED_ORIGINS
        },
        "image_quality": {
            "blur_threshold": BLUR_THRESHOLD,
            "brightness_target": BRIGHTNESS_TARGET,
            "contrast_threshold": CONTRAST_THRESHOLD
        }
    }

def print_config():
    """Print current configuration"""
    config = get_config()
    print("🔧 Bedrock OCR Configuration")
    print("=" * 50)
    
    for section, settings in config.items():
        print(f"\n{section.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    print_config() 