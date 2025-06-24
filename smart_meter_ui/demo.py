#!/usr/bin/env python3
"""
Demo script for Smart Meter Reading System
Tests the FastAPI backend endpoints
"""

import requests
import json
import time
import os
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_database_stats():
    """Test database statistics endpoint"""
    print("\n📊 Testing database statistics...")
    try:
        response = requests.get(f"{BASE_URL}/database/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Database stats: {data}")
            return True
        else:
            print(f"❌ Database stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Database stats error: {e}")
        return False

def test_database_search():
    """Test database search endpoint"""
    print("\n🔍 Testing database search...")
    try:
        response = requests.get(f"{BASE_URL}/database/search?limit=5")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Database search: Found {data['count']} records")
            return True
        else:
            print(f"❌ Database search failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Database search error: {e}")
        return False

def test_image_upload(image_path):
    """Test image upload endpoint"""
    print(f"\n📤 Testing image upload: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Upload successful: {data}")
            return data['image_id']
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

def test_image_processing(image_id):
    """Test image processing endpoint"""
    print(f"\n⚙️ Testing image processing: {image_id}")
    
    try:
        # Start processing
        response = requests.post(f"{BASE_URL}/process/{image_id}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Processing started: {data}")
            
            # Poll for status
            max_attempts = 30  # 30 seconds max
            for attempt in range(max_attempts):
                time.sleep(2)
                
                status_response = requests.get(f"{BASE_URL}/status/{image_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"📊 Status: {status_data['status']}")
                    
                    if status_data['status'] == 'completed':
                        print("✅ Processing completed!")
                        return True
                    elif status_data['status'] == 'failed':
                        print(f"❌ Processing failed: {status_data.get('error')}")
                        return False
                else:
                    print(f"❌ Status check failed: {status_response.status_code}")
                    return False
            
            print("⏰ Processing timeout")
            return False
        else:
            print(f"❌ Processing start failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return False

def test_results(image_id):
    """Test results endpoint"""
    print(f"\n📋 Testing results: {image_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/results/{image_id}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Results retrieved successfully!")
            print(f"📊 Meter Reading: {data['extraction_results']['meter_reading']}")
            print(f"🎯 Confidence: {data['extraction_results']['confidence']}/10")
            print(f"✅ Validation Status: {data['validation_results']['validation_status']}")
            return True
        else:
            print(f"❌ Results failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Results error: {e}")
        return False

def main():
    """Main demo function"""
    print("🚀 Smart Meter Reading System Demo")
    print("=" * 50)
    
    # Test basic endpoints
    if not test_health():
        print("❌ Backend not running. Please start the backend first.")
        return
    
    if not test_database_stats():
        print("❌ Database not accessible.")
        return
    
    if not test_database_search():
        print("❌ Database search failed.")
        return
    
    # Test with sample image
    sample_image = "../000090225651470522210_processed.png"
    if os.path.exists(sample_image):
        print(f"\n🖼️ Testing with sample image: {sample_image}")
        
        # Upload image
        image_id = test_image_upload(sample_image)
        if image_id:
            # Process image
            if test_image_processing(image_id):
                # Get results
                test_results(image_id)
            else:
                print("❌ Image processing failed")
        else:
            print("❌ Image upload failed")
    else:
        print(f"\n⚠️ Sample image not found: {sample_image}")
        print("Skipping image processing test")
    
    print("\n🎉 Demo completed!")
    print("\nTo test the full system:")
    print("1. Start the backend: cd backend && python main.py")
    print("2. Start the frontend: cd frontend && npm start")
    print("3. Open http://localhost:3000 in your browser")

if __name__ == "__main__":
    main() 