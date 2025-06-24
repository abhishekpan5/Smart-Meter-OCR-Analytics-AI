# Amazon Bedrock OCR System

A complete smart meter reading system using Amazon Bedrock's Claude Vision models for OCR processing, with database validation, batch processing, and a FastAPI backend.

## Features

- **Amazon Bedrock Integration**: Uses Claude Vision models for accurate OCR
- **Database Validation**: Validates extracted readings against reference data
- **Batch Processing**: Process up to 100 images simultaneously
- **Processing History**: Track and analyze all processing results
- **Image Quality Assessment**: Automatic confidence adjustment based on image quality
- **Fuzzy Matching**: Hierarchical matching strategies for meter serial numbers
- **FastAPI Backend**: RESTful API with real-time status updates

## Architecture

```
bedrock_ocr/
├── bedrock_ocr.py          # Core OCR processing module
├── bedrock_api.py          # FastAPI backend server
├── database_setup.py       # Database initialization
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. **Install Dependencies**:
   ```bash
   cd bedrock_ocr
   pip install -r requirements.txt
   ```

2. **Configure AWS Credentials**:
   ```bash
   # Set up AWS credentials (choose one method)
   
   # Method 1: AWS CLI
   aws configure
   
   # Method 2: Environment variables
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-east-1
   
   # Method 3: AWS credentials file
   # ~/.aws/credentials
   ```

3. **Initialize Database**:
   ```bash
   python database_setup.py
   ```

## Usage

### Starting the Backend Server

```bash
python bedrock_api.py
```

The server will start on `http://localhost:8001`

### API Endpoints

#### Single Image Processing
- `POST /process` - Process a single image
- `GET /health` - Health check

#### Batch Processing
- `POST /batch/upload` - Upload multiple images
- `POST /batch/process/{batch_id}` - Start batch processing
- `GET /batch/status/{batch_id}` - Get batch status
- `GET /batch/results/{batch_id}` - Get batch results

#### Processing History
- `GET /processing/history` - Get processing history
- `GET /processing/stats` - Get processing statistics
- `GET /processing/history/search` - Search processing history

### Example Usage

#### Process Single Image
```python
import requests

# Upload and process image
with open('meter_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8001/process', files=files)
    result = response.json()
    print(result)
```

#### Batch Processing
```python
import requests

# Upload multiple images
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
    ('files', open('image3.jpg', 'rb'))
]
response = requests.post('http://localhost:8001/batch/upload', files=files)
batch_id = response.json()['batch_id']

# Start processing
requests.post(f'http://localhost:8001/batch/process/{batch_id}')

# Monitor progress
while True:
    status = requests.get(f'http://localhost:8001/batch/status/{batch_id}').json()
    print(f"Progress: {status['progress_percentage']}%")
    if status['status'] == 'completed':
        break

# Get results
results = requests.get(f'http://localhost:8001/batch/results/{batch_id}').json()
print(results)
```

## Configuration

### Model Configuration
The system uses Claude Vision models. You can modify the model in `bedrock_ocr.py`:

```python
bedrock_reader = BedrockOCRReader(
    model_name="anthropic.claude-3-sonnet-20240229-v1:0"  # Change model here
)
```

### Database Configuration
The database path can be configured in the BedrockOCRReader constructor:

```python
bedrock_reader = BedrockOCRReader(
    database_path="path/to/your/database.db"
)
```

## Processing Logic

### 1. Image Quality Assessment
- **Blur Detection**: Uses Laplacian variance to detect image blur
- **Brightness Analysis**: Assesses optimal brightness levels
- **Contrast Evaluation**: Measures text contrast for readability

### 2. OCR Processing
- **Claude Vision**: Extracts meter serial, reading, type, and date
- **Confidence Scoring**: Self-assessed confidence on 1-10 scale
- **Fallback Handling**: Graceful error handling for failed extractions

### 3. Database Validation
- **Exact Match**: Direct serial number match
- **Partial Match**: Substring matching with similarity scoring
- **Fuzzy Match**: Character-based similarity for near matches
- **Reading Validation**: Compares extracted readings against reference data

### 4. Outcome Classification
- **Perfect Match**: Both serial and reading match with high confidence
- **Good Match**: Reading matches but serial doesn't (or vice versa)
- **Partial Match**: Some confidence in either serial or reading
- **No Match**: Low confidence in both

## Database Schema

### meter_reference_data
```sql
CREATE TABLE meter_reference_data (
    meter_serial_number TEXT PRIMARY KEY,
    meter_type TEXT NOT NULL,
    current_reading TEXT NOT NULL,
    installation_date TEXT,
    last_reading_date TEXT
);
```

### processing_results
```sql
CREATE TABLE processing_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    extracted_serial TEXT,
    extracted_reading TEXT,
    extracted_type TEXT,
    extracted_date TEXT,
    manual_serial TEXT,
    manual_reading TEXT,
    manual_type TEXT,
    manual_date TEXT,
    serial_match_type TEXT,
    serial_confidence REAL,
    reading_validated BOOLEAN,
    reading_confidence REAL,
    processing_outcome TEXT,
    ai_confidence REAL,
    extraction_notes TEXT,
    processing_timestamp TEXT
);
```

## Error Handling

The system includes comprehensive error handling:

- **AWS Credentials**: Validates AWS access before processing
- **Image Processing**: Handles corrupted or unsupported image formats
- **API Errors**: Graceful handling of Bedrock API failures
- **Database Errors**: Connection management and retry logic
- **Validation Errors**: Fallback values for failed extractions

## Performance Considerations

- **Concurrency Control**: Limits concurrent API calls to prevent rate limiting
- **Database Indexing**: Optimized queries with proper indexes
- **Memory Management**: Efficient image processing and cleanup
- **Batch Processing**: Background processing with progress tracking

## Monitoring and Logging

The system provides detailed logging for:
- Processing steps and results
- Error conditions and resolutions
- Performance metrics
- Database operations

## Security Considerations

- **AWS IAM**: Use least-privilege access for Bedrock services
- **Input Validation**: File type and size validation
- **Error Sanitization**: Prevents information leakage in error messages
- **Database Security**: Parameterized queries to prevent injection

## Troubleshooting

### Common Issues

1. **AWS Credentials Error**:
   ```
   AWS credentials not found. Please configure AWS credentials.
   ```
   Solution: Configure AWS credentials using AWS CLI or environment variables

2. **Model Not Found**:
   ```
   Model not found or access denied
   ```
   Solution: Verify model name and ensure Bedrock access is enabled

3. **Database Connection Error**:
   ```
   Cannot operate on a closed database
   ```
   Solution: Restart the server to reinitialize database connections

4. **Image Processing Error**:
   ```
   Failed to parse JSON response
   ```
   Solution: Check image quality and ensure it's a valid meter image

### Debug Mode

Enable debug logging by modifying the logging level in the code:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. 