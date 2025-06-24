# 🎯 Smart Meter Reading System

A comprehensive AI-powered smart meter reading system with GPT-4 Vision integration, database validation, and a modern React UI.

## ✨ Features

### 🤖 AI-Powered Reading
- **GPT-4 Vision Integration**: Advanced AI for accurate meter reading extraction
- **Multi-format Support**: JPEG, PNG, BMP, TIFF image formats
- **Confidence Scoring**: Detailed confidence metrics for each extraction
- **Date/Time Detection**: Automatic extraction of meter timestamps

### 🔍 Database Validation
- **Reference Database**: SQLite database with historical meter readings
- **Exact Matching**: Find exact matches in the database
- **Closest Matches**: Identify similar readings with percentage differences
- **Temporal Analysis**: Validate readings against historical patterns

### 📊 Comprehensive Analytics
- **Accuracy Metrics**: Detailed accuracy analysis and error reporting
- **Processing Summary**: Complete processing statistics
- **Recommendations**: AI-generated recommendations for improvement
- **Visual Reports**: Beautiful charts and metrics visualization

### 🎨 Modern UI/UX
- **React Frontend**: Modern, responsive Material-UI interface
- **Drag & Drop Upload**: Intuitive image upload experience
- **Real-time Processing**: Live status updates and progress tracking
- **Detailed Results**: Comprehensive results display with all metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- OpenAI API key

### 1. Setup Environment
```bash
# Clone or navigate to the project
cd smart_meter_ui

# Create and activate virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install Python dependencies
cd backend
pip install -r requirements.txt
cd ..
```

### 2. Configure API Key
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start the System
```bash
# Make the startup script executable (first time only)
chmod +x start.sh

# Start both backend and frontend
./start.sh
```

The system will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📁 Project Structure

```
smart_meter_ui/
├── backend/
│   ├── main.py              # FastAPI backend server
│   ├── requirements.txt     # Python dependencies
│   ├── uploads/            # Uploaded images
│   ├── processed/          # Processed images
│   └── results/            # Processing results
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx
│   │   │   ├── Dashboard.tsx
│   │   │   ├── ImageUpload.tsx
│   │   │   ├── ResultsView.tsx
│   │   │   └── DatabaseView.tsx
│   │   └── App.tsx
│   └── package.json
├── start.sh                # Startup script
└── README.md
```

## 🔧 API Endpoints

### Core Endpoints
- `POST /upload` - Upload meter image
- `POST /process/{image_id}` - Process uploaded image
- `GET /status/{image_id}` - Get processing status
- `GET /results/{image_id}` - Get detailed results

### Database Endpoints
- `GET /database/stats` - Get database statistics
- `GET /database/search` - Search database records

### Health & Info
- `GET /` - API information
- `GET /health` - Health check

## 🎯 Usage Workflow

### 1. Upload Image
- Navigate to the Upload page
- Drag & drop or select a meter image
- Supported formats: JPEG, PNG, BMP, TIFF

### 2. Processing
- System automatically starts GPT-4 Vision processing
- Real-time status updates show progress
- Background validation against database

### 3. View Results
- Comprehensive extraction results
- Database validation status
- Accuracy metrics and recommendations
- Detailed confidence scores

### 4. Database Management
- View database statistics
- Search for specific readings
- Browse historical records

## 📊 Results Analysis

### Extraction Results
- **Meter Reading**: Extracted numerical value
- **Meter Type**: Type of meter detected
- **Display Type**: LCD, LED, mechanical, etc.
- **Confidence Score**: 0-10 confidence rating
- **Date/Time**: Extracted timestamp with confidence

### Validation Results
- **Validation Status**: Match/No Match/Partial
- **Database Matches**: Exact matches found
- **Closest Matches**: Similar readings with differences
- **Temporal Validation**: Historical pattern analysis

### Accuracy Metrics
- **Exact Accuracy**: Percentage of exact matches
- **Close Accuracy**: Percentage within acceptable range
- **Average Confidence**: Overall confidence score
- **Mean Error**: Average percentage error

## 🔍 Database Schema

```sql
CREATE TABLE meter_readings (
    id INTEGER PRIMARY KEY,
    meter_serial_number INTEGER,
    meter_reading REAL,
    original_reading TEXT,
    reading_unit TEXT,
    reading_date TEXT
);
```

## 🛠️ Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend Development
```bash
cd frontend
npm install
npm start
```

### Adding New Features
1. **Backend**: Add new endpoints in `main.py`
2. **Frontend**: Create new components in `src/components/`
3. **Database**: Update schema and validation logic

## 🔒 Security Considerations

- API key stored in environment variables
- File upload validation and sanitization
- CORS configuration for frontend access
- Input validation on all endpoints

## 📈 Performance Optimization

- Background processing for long-running tasks
- Image compression and optimization
- Database indexing for fast searches
- Caching for frequently accessed data

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Error**
   - Check API key in `.env` file
   - Verify API key has GPT-4 Vision access
   - Check API quota and billing

2. **Database Connection Error**
   - Ensure `smart_meter_database.db` exists
   - Check file permissions
   - Verify database schema

3. **Image Upload Issues**
   - Check file format (JPEG, PNG, BMP, TIFF)
   - Verify file size (max 10MB)
   - Ensure image is clear and well-lit

4. **Processing Failures**
   - Check GPT-4 Vision API status
   - Verify image quality and format
   - Review error logs in backend

### Logs
- Backend logs: Check terminal output
- Frontend logs: Browser developer console
- API errors: Check FastAPI docs at `/docs`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT-4 Vision API
- Material-UI for the beautiful React components
- FastAPI for the robust backend framework
- React community for excellent tooling

---

**Ready to revolutionize your meter reading process? Start the system and upload your first image! 🚀** 