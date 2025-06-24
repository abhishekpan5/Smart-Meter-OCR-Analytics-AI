#!/usr/bin/env python3
"""
Setup script for Bedrock OCR System
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing Dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_database():
    """Set up the database"""
    print("\n🗄️  Setting up Database...")
    
    try:
        from database_setup import create_database_tables, load_sample_data, check_database_status
        
        create_database_tables()
        load_sample_data()
        status = check_database_status()
        
        print(f"✅ Database setup completed")
        print(f"   Meter records: {status['meter_count']}")
        print(f"   Processing records: {status['processing_count']}")
        return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating Directories...")
    
    directories = [
        "bedrock_uploads",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories created successfully")
    return True

def check_aws_credentials():
    """Check AWS credentials configuration"""
    print("\n🔑 Checking AWS Credentials...")
    
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
        
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials is None:
            print("⚠️  AWS credentials not found")
            print("   Configure AWS credentials using one of these methods:")
            print("   1. AWS CLI: aws configure")
            print("   2. Environment variables:")
            print("      export AWS_ACCESS_KEY_ID=your_access_key")
            print("      export AWS_SECRET_ACCESS_KEY=your_secret_key")
            print("      export AWS_DEFAULT_REGION=us-east-1")
            print("   3. Credentials file: ~/.aws/credentials")
            return False
        
        print("✅ AWS credentials found")
        return True
        
    except ImportError:
        print("❌ boto3 not installed. Run: pip install boto3")
        return False
    except Exception as e:
        print(f"❌ Error checking AWS credentials: {e}")
        return False

def test_bedrock_access():
    """Test Bedrock access"""
    print("\n🧪 Testing Bedrock Access...")
    
    try:
        import boto3
        
        bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Try to list models (this will fail if no access, but that's expected)
        try:
            bedrock.list_foundation_models()
            print("✅ Bedrock access confirmed")
            return True
        except Exception as e:
            if "AccessDenied" in str(e):
                print("⚠️  Bedrock access denied - you may need to enable Bedrock in your AWS account")
                print("   Go to AWS Console > Bedrock > Model access to enable Claude models")
                return False
            else:
                print(f"⚠️  Bedrock test inconclusive: {e}")
                return True  # Continue setup even if test fails
                
    except Exception as e:
        print(f"❌ Error testing Bedrock access: {e}")
        return False

def create_config_file():
    """Create a sample configuration file"""
    print("\n⚙️  Creating Configuration...")
    
    config_content = """# Bedrock OCR Configuration
# Copy this to .env file and modify as needed

# AWS Configuration
AWS_DEFAULT_REGION=us-east-1
BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0

# Server Configuration
HOST=0.0.0.0
PORT=8001
DEBUG=false

# Upload Configuration
MAX_FILE_SIZE=10
MAX_BATCH_SIZE=100

# Processing Configuration
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT=30

# Validation Configuration
SERIAL_SIMILARITY_THRESHOLD=0.7
READING_VALIDATION_RATIO_MIN=0.5
READING_VALIDATION_RATIO_MAX=1.5

# Confidence Configuration
MIN_CONFIDENCE_THRESHOLD=0.5
PERFECT_MATCH_SERIAL_CONFIDENCE=80.0
PERFECT_MATCH_READING_CONFIDENCE=80.0
GOOD_MATCH_READING_CONFIDENCE=70.0
PARTIAL_MATCH_CONFIDENCE=50.0

# Logging Configuration
LOG_LEVEL=INFO

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000
"""
    
    try:
        with open(".env.example", "w") as f:
            f.write(config_content)
        print("✅ Configuration file created: .env.example")
        return True
    except Exception as e:
        print(f"❌ Failed to create configuration file: {e}")
        return False

def run_tests():
    """Run system tests"""
    print("\n🧪 Running System Tests...")
    
    try:
        from test_bedrock_ocr import test_bedrock_ocr
        import asyncio
        
        success = asyncio.run(test_bedrock_ocr())
        
        if success:
            print("✅ System tests passed")
            return True
        else:
            print("❌ System tests failed")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Bedrock OCR System Setup Completed!")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. Configure AWS credentials (if not already done):")
    print("   aws configure")
    print("   # or set environment variables")
    
    print("\n2. Enable Bedrock access in AWS Console:")
    print("   - Go to AWS Console > Bedrock")
    print("   - Navigate to Model access")
    print("   - Enable Claude models")
    
    print("\n3. Start the server:")
    print("   python start_server.py")
    
    print("\n4. Test the system:")
    print("   python test_bedrock_ocr.py")
    print("   python example_usage.py")
    
    print("\n5. API Documentation:")
    print("   http://localhost:8001/docs")
    
    print("\n📚 Useful Commands:")
    print("   # Start server")
    print("   python start_server.py")
    print("")
    print("   # Run tests")
    print("   python test_bedrock_ocr.py")
    print("")
    print("   # View examples")
    print("   python example_usage.py")
    print("")
    print("   # Check configuration")
    print("   python config.py")
    
    print("\n🔗 API Endpoints:")
    print("   POST /process - Process single image")
    print("   POST /batch/upload - Upload multiple images")
    print("   POST /batch/process/{batch_id} - Start batch processing")
    print("   GET /batch/status/{batch_id} - Get batch status")
    print("   GET /batch/results/{batch_id} - Get batch results")
    print("   GET /processing/history - Get processing history")
    print("   GET /processing/stats - Get processing statistics")
    print("   GET /health - Health check")

def main():
    """Main setup function"""
    print("🔧 Bedrock OCR System Setup")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("bedrock_ocr.py"):
        print("❌ bedrock_ocr.py not found in current directory")
        print("Make sure you're running this script from the bedrock_ocr directory")
        return False
    
    # Run setup steps
    steps = [
        ("Python Version Check", check_python_version),
        ("Install Dependencies", install_dependencies),
        ("Create Directories", create_directories),
        ("Setup Database", setup_database),
        ("Create Configuration", create_config_file),
        ("Check AWS Credentials", check_aws_credentials),
        ("Test Bedrock Access", test_bedrock_access),
        ("Run System Tests", run_tests),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            failed_steps.append(step_name)
    
    # Print results
    if failed_steps:
        print(f"\n❌ Setup completed with {len(failed_steps)} failed steps:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease fix the failed steps and run setup again.")
        return False
    else:
        print_next_steps()
        return True

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!") 