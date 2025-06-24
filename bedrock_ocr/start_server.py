#!/usr/bin/env python3
"""
Startup script for Bedrock OCR API server
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn', 
        'boto3',
        'opencv-python',
        'Pillow',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_aws_credentials():
    """Check if AWS credentials are configured"""
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
        
        # Try to create a session
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials is None:
            print("❌ AWS credentials not found")
            print("Configure AWS credentials using one of these methods:")
            print("1. AWS CLI: aws configure")
            print("2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            print("3. Credentials file: ~/.aws/credentials")
            return False
        
        print("✅ AWS credentials found")
        return True
        
    except Exception as e:
        print(f"❌ Error checking AWS credentials: {e}")
        return False

def check_database():
    """Check if database exists and has required tables"""
    try:
        from database_setup import check_database_status
        
        status = check_database_status()
        
        if status['meter_count'] == 0:
            print("⚠️  Database exists but has no meter reference data")
            print("Run: python database_setup.py to load sample data")
            return False
        
        print(f"✅ Database ready: {status['meter_count']} meter records")
        return True
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        print("Run: python database_setup.py to initialize database")
        return False

def create_upload_directories():
    """Create necessary upload directories"""
    directories = [
        "bedrock_uploads",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Upload directories created")

def start_server():
    """Start the Bedrock OCR API server"""
    print("\n🚀 Starting Bedrock OCR API Server...")
    print("=" * 50)
    
    # Check prerequisites
    print("\n1. Checking Dependencies...")
    if not check_dependencies():
        return False
    
    print("\n2. Checking AWS Credentials...")
    if not check_aws_credentials():
        return False
    
    print("\n3. Checking Database...")
    if not check_database():
        return False
    
    print("\n4. Creating Directories...")
    create_upload_directories()
    
    # Start the server
    print("\n5. Starting Server...")
    print("   Server will be available at: http://localhost:8001")
    print("   API Documentation: http://localhost:8001/docs")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Import and run the server
        from bedrock_api import app
        import uvicorn
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info",
            reload=True
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🔧 Bedrock OCR Server Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("bedrock_api.py"):
        print("❌ bedrock_api.py not found in current directory")
        print("Make sure you're running this script from the bedrock_ocr directory")
        return False
    
    # Start the server
    return start_server()

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n❌ Server startup failed. Check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ Server startup completed successfully!") 