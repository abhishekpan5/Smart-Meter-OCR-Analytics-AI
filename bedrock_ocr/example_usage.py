#!/usr/bin/env python3
"""
Example usage of Bedrock OCR System
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bedrock_ocr import BedrockOCRReader
from database_setup import create_database_tables, load_sample_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_single_image_processing():
    """Example of processing a single image"""
    print("\n📸 Example: Single Image Processing")
    print("=" * 40)
    
    try:
        # Initialize Bedrock OCR reader
        bedrock_reader = BedrockOCRReader()
        
        # Example image path (replace with actual image)
        image_path = "example_meter.jpg"
        
        if not os.path.exists(image_path):
            print(f"⚠️  Example image not found: {image_path}")
            print("   Create an example_meter.jpg file to test processing")
            return
        
        print(f"Processing image: {image_path}")
        
        # Process the image
        result = await bedrock_reader.process_image(image_path)
        
        # Display results
        print("\n📊 Processing Results:")
        print(f"  Extracted Serial: {result['extracted_data']['meter_serial_number']}")
        print(f"  Extracted Reading: {result['extracted_data']['meter_reading']}")
        print(f"  Extracted Type: {result['extracted_data']['meter_type']}")
        print(f"  Manual Serial: {result['manual_data']['meter_serial_number']}")
        print(f"  Manual Reading: {result['manual_data']['meter_reading']}")
        print(f"  Serial Match Type: {result['validation']['serial_match_type']}")
        print(f"  Serial Confidence: {result['validation']['serial_confidence']:.1f}%")
        print(f"  Reading Validated: {result['validation']['reading_validated']}")
        print(f"  Reading Confidence: {result['validation']['reading_confidence']:.1f}%")
        print(f"  Processing Outcome: {result['validation']['processing_outcome']}")
        print(f"  AI Confidence: {result['ai_confidence']:.2f}")
        print(f"  Extraction Notes: {result['extraction_notes']}")
        
        # Save to database
        if bedrock_reader.save_processing_result(image_path, result):
            print("✅ Results saved to database")
        else:
            print("❌ Failed to save results to database")
            
    except Exception as e:
        print(f"❌ Error in single image processing: {e}")

async def example_batch_processing():
    """Example of batch processing multiple images"""
    print("\n📦 Example: Batch Processing")
    print("=" * 40)
    
    try:
        # Initialize Bedrock OCR reader
        bedrock_reader = BedrockOCRReader()
        
        # Example image paths (replace with actual images)
        image_paths = [
            "example_meter1.jpg",
            "example_meter2.jpg", 
            "example_meter3.jpg"
        ]
        
        # Filter to only existing images
        existing_images = [path for path in image_paths if os.path.exists(path)]
        
        if not existing_images:
            print("⚠️  No example images found")
            print("   Create example_meter1.jpg, example_meter2.jpg, etc. to test batch processing")
            return
        
        print(f"Processing {len(existing_images)} images: {existing_images}")
        
        # Process batch
        batch_result = await bedrock_reader.process_batch(existing_images)
        
        # Display batch results
        print(f"\n📊 Batch Results:")
        print(f"  Total Images: {batch_result['total_images']}")
        print(f"  Successful: {batch_result['successful']}")
        print(f"  Failed: {batch_result['failed']}")
        
        # Show individual results
        for i, item in enumerate(batch_result['results']):
            if item['status'] == 'success':
                result = item['result']
                print(f"\n  Image {i+1}: {os.path.basename(item['image_path'])}")
                print(f"    Outcome: {result['validation']['processing_outcome']}")
                print(f"    Serial: {result['extracted_data']['meter_serial_number']}")
                print(f"    Reading: {result['extracted_data']['meter_reading']}")
            else:
                print(f"\n  Image {i+1}: {os.path.basename(item['image_path'])} - FAILED")
                
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")

def example_database_operations():
    """Example of database operations"""
    print("\n🗄️  Example: Database Operations")
    print("=" * 40)
    
    try:
        # Initialize Bedrock OCR reader
        bedrock_reader = BedrockOCRReader()
        
        # Get processing history
        history = bedrock_reader.get_processing_history(limit=5)
        print(f"📋 Recent Processing History:")
        print(f"  Total Records: {history['total_count']}")
        print(f"  Showing: {len(history['history'])} records")
        
        for i, record in enumerate(history['history']):
            print(f"\n  Record {i+1}:")
            print(f"    Image: {os.path.basename(record['image_path'])}")
            print(f"    Outcome: {record['processing_outcome']}")
            print(f"    Extracted Serial: {record['extracted_serial']}")
            print(f"    Extracted Reading: {record['extracted_reading']}")
            print(f"    Timestamp: {record['processing_timestamp']}")
        
        # Get processing statistics
        stats = bedrock_reader.get_processing_stats()
        print(f"\n📈 Processing Statistics:")
        print(f"  Total Records: {stats['total_records']}")
        print(f"  Average Confidence: {stats['average_confidence']:.2f}")
        print(f"  Recent Activity (24h): {stats['recent_activity_24h']}")
        
        if stats['outcome_distribution']:
            print(f"  Outcome Distribution:")
            for outcome, count in stats['outcome_distribution'].items():
                print(f"    {outcome}: {count}")
                
    except Exception as e:
        print(f"❌ Error in database operations: {e}")

def example_fuzzy_matching():
    """Example of fuzzy matching functionality"""
    print("\n🔍 Example: Fuzzy Matching")
    print("=" * 40)
    
    try:
        # Initialize Bedrock OCR reader
        bedrock_reader = BedrockOCRReader()
        
        # Test cases
        test_cases = [
            "12345678",  # Exact match (if exists in database)
            "123456",    # Partial match
            "123456789", # Similar to existing
            "99999999",  # No match
            "Not visible", # Invalid input
        ]
        
        print("Testing fuzzy matching with different inputs:")
        
        for test_serial in test_cases:
            match, match_type, confidence = bedrock_reader._find_best_match(test_serial)
            
            if match:
                print(f"  '{test_serial}' -> {match_type} ({confidence:.1f}%) -> {match['meter_serial_number']}")
            else:
                print(f"  '{test_serial}' -> {match_type} ({confidence:.1f}%) -> No match")
                
    except Exception as e:
        print(f"❌ Error in fuzzy matching: {e}")

def example_reading_validation():
    """Example of reading validation"""
    print("\n✅ Example: Reading Validation")
    print("=" * 40)
    
    try:
        # Initialize Bedrock OCR reader
        bedrock_reader = BedrockOCRReader()
        
        # Get a reference reading from database
        if bedrock_reader.reference_data:
            ref_serial = list(bedrock_reader.reference_data.keys())[0]
            ref_reading = bedrock_reader.reference_data[ref_serial]['current_reading']
            
            print(f"Reference reading: {ref_reading}")
            
            # Test cases
            test_cases = [
                ref_reading,           # Exact match
                str(float(ref_reading) + 100),  # Close match
                str(float(ref_reading) * 2),    # Too high
                str(float(ref_reading) * 0.3),  # Too low
                "99999.99",            # Invalid
                "Not visible",         # Invalid
            ]
            
            print("\nTesting reading validation:")
            
            for test_reading in test_cases:
                validated, confidence = bedrock_reader._validate_reading(test_reading, ref_reading)
                print(f"  '{test_reading}' -> Valid: {validated}, Confidence: {confidence:.1f}%")
        else:
            print("⚠️  No reference data available for testing")
            
    except Exception as e:
        print(f"❌ Error in reading validation: {e}")

async def main():
    """Main example function"""
    print("🚀 Bedrock OCR System - Example Usage")
    print("=" * 50)
    
    # Ensure database is set up
    print("\n1. Setting up database...")
    try:
        create_database_tables()
        load_sample_data()
        print("✅ Database setup completed")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return
    
    # Run examples
    await example_single_image_processing()
    await example_batch_processing()
    example_database_operations()
    example_fuzzy_matching()
    example_reading_validation()
    
    print("\n" + "=" * 50)
    print("🎉 Example usage completed!")
    print("\nNext steps:")
    print("1. Add actual meter images to test processing")
    print("2. Configure AWS credentials for Bedrock access")
    print("3. Start the API server: python start_server.py")
    print("4. Use the API endpoints for integration")

if __name__ == "__main__":
    asyncio.run(main()) 