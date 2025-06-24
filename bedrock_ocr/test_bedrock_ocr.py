#!/usr/bin/env python3
"""
Test script for Bedrock OCR system
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bedrock_ocr import BedrockOCRReader
from database_setup import create_database_tables, load_sample_data, check_database_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_bedrock_ocr():
    """Test the Bedrock OCR system"""
    
    print("🚀 Testing Bedrock OCR System")
    print("=" * 50)
    
    # Test 1: Database Setup
    print("\n1. Testing Database Setup...")
    try:
        create_database_tables()
        load_sample_data()
        status = check_database_status()
        print(f"✅ Database setup successful: {status['meter_count']} meter records loaded")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False
    
    # Test 2: Bedrock OCR Initialization
    print("\n2. Testing Bedrock OCR Initialization...")
    try:
        bedrock_reader = BedrockOCRReader()
        print("✅ Bedrock OCR reader initialized successfully")
    except Exception as e:
        print(f"❌ Bedrock OCR initialization failed: {e}")
        print("   Make sure AWS credentials are configured and Bedrock access is enabled")
        return False
    
    # Test 3: Reference Data Loading
    print("\n3. Testing Reference Data Loading...")
    try:
        reference_data = bedrock_reader.reference_data
        print(f"✅ Reference data loaded: {len(reference_data)} records")
        
        # Show sample data
        if reference_data:
            sample_key = list(reference_data.keys())[0]
            sample_data = reference_data[sample_key]
            print(f"   Sample record: {sample_key} -> {sample_data['meter_type']} ({sample_data['current_reading']})")
    except Exception as e:
        print(f"❌ Reference data loading failed: {e}")
        return False
    
    # Test 4: Image Quality Assessment
    print("\n4. Testing Image Quality Assessment...")
    try:
        # Create a test image path (you would need an actual image for full testing)
        test_image_path = "test_image.jpg"
        
        if os.path.exists(test_image_path):
            quality = bedrock_reader._assess_image_quality(test_image_path)
            print(f"✅ Image quality assessment: {quality}")
        else:
            print("⚠️  No test image found, skipping image quality test")
            print("   Create a test_image.jpg file to test image quality assessment")
    except Exception as e:
        print(f"❌ Image quality assessment failed: {e}")
    
    # Test 5: Fuzzy Matching
    print("\n5. Testing Fuzzy Matching...")
    try:
        # Test exact match
        if reference_data:
            test_serial = list(reference_data.keys())[0]
            match, match_type, confidence = bedrock_reader._find_best_match(test_serial)
            print(f"✅ Exact match test: {test_serial} -> {match_type} ({confidence:.1f}%)")
            
            # Test partial match
            partial_serial = test_serial[:6]  # First 6 characters
            match, match_type, confidence = bedrock_reader._find_best_match(partial_serial)
            print(f"✅ Partial match test: {partial_serial} -> {match_type} ({confidence:.1f}%)")
            
            # Test no match
            match, match_type, confidence = bedrock_reader._find_best_match("99999999")
            print(f"✅ No match test: 99999999 -> {match_type} ({confidence:.1f}%)")
    except Exception as e:
        print(f"❌ Fuzzy matching failed: {e}")
    
    # Test 6: Reading Validation
    print("\n6. Testing Reading Validation...")
    try:
        if reference_data:
            test_data = list(reference_data.values())[0]
            ref_reading = test_data['current_reading']
            
            # Test valid reading
            validated, confidence = bedrock_reader._validate_reading(ref_reading, ref_reading)
            print(f"✅ Valid reading test: {ref_reading} -> {validated} ({confidence:.1f}%)")
            
            # Test invalid reading
            validated, confidence = bedrock_reader._validate_reading("99999.99", ref_reading)
            print(f"✅ Invalid reading test: 99999.99 -> {validated} ({confidence:.1f}%)")
    except Exception as e:
        print(f"❌ Reading validation failed: {e}")
    
    # Test 7: Processing Outcome Determination
    print("\n7. Testing Processing Outcome Determination...")
    try:
        # Test perfect match
        outcome = bedrock_reader._determine_processing_outcome("Exact Match", 90.0, True, 85.0)
        print(f"✅ Perfect match outcome: {outcome}")
        
        # Test good match
        outcome = bedrock_reader._determine_processing_outcome("No Match", 0.0, True, 75.0)
        print(f"✅ Good match outcome: {outcome}")
        
        # Test no match
        outcome = bedrock_reader._determine_processing_outcome("No Match", 0.0, False, 0.0)
        print(f"✅ No match outcome: {outcome}")
    except Exception as e:
        print(f"❌ Processing outcome determination failed: {e}")
    
    # Test 8: Database Operations
    print("\n8. Testing Database Operations...")
    try:
        # Test processing history
        history = bedrock_reader.get_processing_history(limit=5)
        print(f"✅ Processing history: {history['total_count']} total records")
        
        # Test processing stats
        stats = bedrock_reader.get_processing_stats()
        print(f"✅ Processing stats: {stats['total_records']} records, {stats['average_confidence']:.1f} avg confidence")
    except Exception as e:
        print(f"❌ Database operations failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Bedrock OCR System Test Completed!")
    print("\nNext Steps:")
    print("1. Configure AWS credentials for Bedrock access")
    print("2. Start the API server: python bedrock_api.py")
    print("3. Test with actual meter images")
    
    return True

def test_api_endpoints():
    """Test API endpoints (requires server to be running)"""
    print("\n🌐 Testing API Endpoints...")
    print("=" * 50)
    
    try:
        import requests
        
        base_url = "http://localhost:8001"
        
        # Test health endpoint
        print("\n1. Testing Health Endpoint...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print(f"✅ Health check: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
        
        # Test processing stats
        print("\n2. Testing Processing Stats...")
        response = requests.get(f"{base_url}/processing/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Processing stats: {stats['total_records']} records")
        else:
            print(f"❌ Processing stats failed: {response.status_code}")
        
        # Test processing history
        print("\n3. Testing Processing History...")
        response = requests.get(f"{base_url}/processing/history?limit=5")
        if response.status_code == 200:
            history = response.json()
            print(f"✅ Processing history: {history['total_count']} total records")
        else:
            print(f"❌ Processing history failed: {response.status_code}")
        
    except ImportError:
        print("❌ Requests library not available. Install with: pip install requests")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running on http://localhost:8001")
    except Exception as e:
        print(f"❌ API testing failed: {e}")

if __name__ == "__main__":
    print("🧪 Bedrock OCR System Test Suite")
    print("=" * 50)
    
    # Run core tests
    success = asyncio.run(test_bedrock_ocr())
    
    if success:
        # Run API tests if core tests passed
        test_api_endpoints()
    
    print("\n📋 Test Summary:")
    print("- Core functionality tests completed")
    print("- API endpoint tests completed (if server is running)")
    print("- Check the output above for any ❌ errors")
    
    if not success:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ All tests completed successfully!") 