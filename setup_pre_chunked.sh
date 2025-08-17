#!/bin/bash


python -m venv venv
source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi


#start docker containers
docker compose up -d

# Wait for container to be ready
echo "⏳ Waiting for Ollama to start..."
sleep 10

# Check if container is running
if ! docker ps | grep -q ollama; then
    echo "❌ Failed to start Ollama container"
    exit 1
fi

echo "✅ Ollama container is running"

# Download model 
echo "📥 Downloading llama3.2:1b model (this may take a few minutes)..."
docker exec -it ollama ollama pull llama3.2:1b

#embed godot docs
python embed.py

#import monitoring dashboard for grafana
./import_dashboard.sh

streamlit run app.py

