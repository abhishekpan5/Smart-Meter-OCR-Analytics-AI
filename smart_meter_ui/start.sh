#!/bin/bash

echo "🚀 Starting Smart Meter Reading System..."

# Check if virtual environment exists
if [ ! -d "../myenv" ]; then
    echo "❌ Virtual environment not found. Please run 'python -m venv myenv' and activate it first."
    exit 1
fi

# Check if .env file exists
if [ ! -f "../.env" ]; then
    echo "⚠️  .env file not found. Please create one with your OpenAI API key:"
    echo "OPENAI_API_KEY=your_api_key_here"
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "📦 Installing backend dependencies..."
cd backend
pip install -r requirements.txt

echo "🔧 Starting FastAPI backend server..."
python main.py &
BACKEND_PID=$!

echo "⏳ Waiting for backend to start..."
sleep 5

echo "📦 Installing frontend dependencies..."
cd ../frontend
npm install

echo "🎨 Starting React frontend server..."
npm start &
FRONTEND_PID=$!

echo "✅ Smart Meter Reading System is starting up!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔌 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for both processes
wait 