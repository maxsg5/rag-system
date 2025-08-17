#!/bin/bash

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

# Test the setup
echo "🧪 Testing Ollama setup..."
response=$(docker exec ollama ollama run llama3.2:1b "Say hello!" --timeout 30)

if [ $? -eq 0 ]; then
    echo "✅ Ollama setup complete!"
    echo "📝 Model response: $response"
    echo ""
    echo "🌐 Ollama is running at: http://localhost:11434"
    echo "📚 See README.md for usage examples"
    echo ""
    echo "Quick test commands:"
    echo "  docker exec -it ollama ollama run llama3.2:1b \"Your question here\""
    echo "  curl -X POST http://localhost:11434/api/generate -d '{\"model\": \"llama3.2:1b\", \"prompt\": \"Hello!\", \"stream\": false}'"
else
    echo "❌ Setup test failed. Check container logs:"
    echo "  docker logs ollama"
fi


# Download godot docs
python download_godot_docs.py

# chunk godot docs
python test_chunk.py

#embed godot docs
python test_embed.py




