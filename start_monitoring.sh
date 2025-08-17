#!/bin/bash

echo "🚀 Starting RAG System Monitoring"

# Check service health
echo "🔍 Checking service health..."

# Check Qdrant
if curl -s http://localhost:6333/healthz > /dev/null; then
    echo "✅ Qdrant is ready"
else
    echo "❌ Qdrant not ready"
fi

# Check Ollama
if curl -s http://localhost:11434/api/ps > /dev/null; then
    echo "✅ Ollama is ready"
else
    echo "❌ Ollama not ready"
fi

# Check Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus is ready"
else
    echo "❌ Prometheus not ready"
fi

# Check Grafana
if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✅ Grafana is ready"
else
    echo "❌ Grafana not ready"
fi

# Check Metrics Exporter
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Metrics Exporter is ready"
else
    echo "❌ Metrics Exporter not ready"
fi

echo ""
echo "🌐 Service URLs:"
echo "  📊 Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "  🔍 Prometheus: http://localhost:9090"
echo "  🗃️ Qdrant: http://localhost:6333"
echo "  🤖 Ollama: http://localhost:11434"
echo "  📈 Metrics: http://localhost:8000/metrics"
echo ""
echo "🚀 Ready to start Streamlit app with:"
echo "  streamlit run app.py"
