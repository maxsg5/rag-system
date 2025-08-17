#!/bin/bash
#
# RAG System Dashboard Import Script
# 
# This script fixes common Grafana datasource UID issues and imports the dashboard
# with the latest configuration including:
# - User Satisfaction Rate (instead of Average Response Time) 
# - User Feedback Metrics panel
# - Proper datasource UID consistency ("prometheus")
#
# Issues this script resolves:
# 1. Grafana auto-generates datasource UIDs that don't match dashboard JSON
# 2. Old dashboard configurations cached causing "datasource not found" errors  
# 3. Manual panel editing required to refresh datasource references
#
# Usage: ./import_dashboard.sh
# Prerequisites: Grafana running on localhost:3000, Python 3 available
#

echo "📊 Importing RAG System Dashboard to Grafana..."

# Wait a moment to ensure Grafana is ready
sleep 2

# Function to check if Grafana is ready
check_grafana_ready() {
    for i in {1..30}; do
        if curl -s http://admin:admin@localhost:3000/api/health >/dev/null 2>&1; then
            echo "✅ Grafana is ready"
            return 0
        fi
        echo "⏳ Waiting for Grafana to be ready... (attempt $i/30)"
        sleep 2
    done
    echo "❌ Grafana failed to become ready"
    exit 1
}

check_grafana_ready

# Step 1: Clean up any existing Prometheus datasources to avoid UID conflicts
echo "🧹 Cleaning up existing Prometheus datasources..."
EXISTING_DS=$(curl -s -u admin:admin http://localhost:3000/api/datasources)
echo "$EXISTING_DS" | python3 -c "
import json, sys, requests
try:
    data = json.load(sys.stdin)
    for ds in data:
        if ds.get('type') == 'prometheus':
            uid = ds.get('uid', '')
            ds_id = ds.get('id', '')
            print(f'Deleting existing Prometheus datasource: {ds.get(\"name\", \"Unknown\")} (UID: {uid})')
            # Delete by UID if available, otherwise by ID
            if uid:
                requests.delete(f'http://admin:admin@localhost:3000/api/datasources/uid/{uid}')
            elif ds_id:
                requests.delete(f'http://admin:admin@localhost:3000/api/datasources/{ds_id}')
except: pass
" 2>/dev/null || true

# Step 2: Create Prometheus datasource with specific UID that matches dashboard
echo "🔗 Creating Prometheus datasource with correct UID..."
DATASOURCE_RESULT=$(curl -s -X POST \
  -u admin:admin \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Prometheus",
    "type": "prometheus", 
    "url": "http://prometheus:9090",
    "access": "proxy",
    "uid": "prometheus",
    "isDefault": true,
    "editable": true
  }' \
  http://localhost:3000/api/datasources)

if echo "$DATASOURCE_RESULT" | grep -q '"uid":"prometheus"'; then
    echo "✅ Prometheus datasource created with correct UID: prometheus"
elif echo "$DATASOURCE_RESULT" | grep -q "already exists"; then
    echo "ℹ️  Prometheus datasource already exists, verifying UID..."
    # Verify the UID is correct
    CURRENT_DS=$(curl -s -u admin:admin http://localhost:3000/api/datasources/name/Prometheus)
    CURRENT_UID=$(echo "$CURRENT_DS" | python3 -c "import json, sys; print(json.load(sys.stdin).get('uid', 'N/A'))" 2>/dev/null || echo "N/A")
    if [ "$CURRENT_UID" = "prometheus" ]; then
        echo "✅ Existing datasource has correct UID: prometheus"
    else
        echo "⚠️  Existing datasource has wrong UID: $CURRENT_UID, expected: prometheus"
        echo "🔄 This may cause dashboard issues. Consider restarting the script."
    fi
else
    echo "⚠️  Datasource creation response: $DATASOURCE_RESULT"
fi

# Step 3: Wait a moment for datasource to be fully ready
sleep 2

# Step 4: Delete existing dashboard if it exists to force clean import
echo "🗑️  Removing existing dashboard to ensure clean import..."
curl -s -X DELETE -u admin:admin http://localhost:3000/api/dashboards/uid/rag_dashboard >/dev/null 2>&1 || true

# Step 5: Import the dashboard
echo "📈 Importing dashboard with User Satisfaction Rate and Feedback Metrics..."
RESULT=$(curl -s -X POST \
  -u admin:admin \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/dashboard_import.json \
  http://localhost:3000/api/dashboards/db)

if echo "$RESULT" | grep -q "success\|id"; then
    echo "✅ Dashboard imported successfully!"
    
    # Step 6: Verify dashboard panels are using correct datasource
    echo "🔍 Verifying dashboard configuration..."
    DASHBOARD_CHECK=$(curl -s -u admin:admin "http://localhost:3000/api/dashboards/uid/rag_dashboard")
    PANEL_COUNT=$(echo "$DASHBOARD_CHECK" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    panels = data.get('dashboard', {}).get('panels', [])
    print(f'Found {len(panels)} panels:')
    for panel in panels:
        title = panel.get('title', 'Unknown')
        ds_uid = panel.get('datasource', {}).get('uid', 'N/A')
        print(f'  - {title}: datasource UID = {ds_uid}')
    # Check for our key panels
    titles = [p.get('title', '') for p in panels]
    if 'User Satisfaction Rate' in titles:
        print('✅ User Satisfaction Rate panel found')
    if 'User Feedback Metrics' in titles:
        print('✅ User Feedback Metrics panel found')
    if 'Average Response Time' in titles:
        print('⚠️  Old Average Response Time panel still present')
except Exception as e:
    print(f'Panel verification failed: {e}')
" 2>/dev/null || echo "Panel verification skipped")
    
    echo "🌐 Access your dashboard at: http://localhost:3000/d/rag_dashboard"
else
    echo "❌ Dashboard import failed!"
    echo "📋 Response: $RESULT"
    echo "🔧 Troubleshooting steps:"
    echo "   1. Check if Grafana is running: docker ps | grep grafana"
    echo "   2. Check Grafana logs: docker logs grafana"
    echo "   3. Manually import: http://localhost:3000 → + → Import → paste monitoring/grafana/dashboard_import.json"
    exit 1
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
