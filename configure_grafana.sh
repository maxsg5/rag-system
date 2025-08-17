#!/bin/bash

echo "🔧 Configuring Grafana automatically..."

# Wait for Grafana to be ready
echo "⏳ Waiting for Grafana to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
        echo "✅ Grafana is ready!"
        break
    fi
    sleep 2
done

# Wait a bit more for full initialization
sleep 5

echo "📊 Creating Prometheus datasource..."
# Create Prometheus datasource
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://prometheus:9090",
    "access": "proxy",
    "isDefault": true
  }' \
  http://admin:admin@localhost:3000/api/datasources

echo ""
echo "📈 Importing RAG System dashboard..."
# Import dashboard
curl -X POST \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/dashboard_import.json \
  http://admin:admin@localhost:3000/api/dashboards/db

echo ""
echo "✅ Grafana configuration complete!"
echo "🌐 Access Grafana at: http://localhost:3000 (admin/admin)"
echo "📊 RAG System Dashboard should be available in the General folder"
