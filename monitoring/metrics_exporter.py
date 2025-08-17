#!/usr/bin/env python3

import os
import time
import requests
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry, generate_latest
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import uvicorn
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create custom registry
registry = CollectorRegistry()

# Metrics
rag_queries_total = Counter('rag_queries_total', 'Total number of RAG queries', registry=registry)
rag_response_time_seconds = Histogram('rag_response_time_seconds', 'Response time for RAG queries', registry=registry)
rag_vector_searches_total = Counter('rag_vector_searches_total', 'Total vector searches performed', registry=registry)
rag_query_types_total = Counter('rag_query_types_total', 'Query types processed', ['query_type'], registry=registry)

# LLM Evaluation Metrics
rag_llm_evaluation_relevance = Gauge('rag_llm_evaluation_relevance', 'LLM evaluation relevance score', registry=registry)
rag_llm_evaluation_accuracy = Gauge('rag_llm_evaluation_accuracy', 'LLM evaluation accuracy score', registry=registry)
rag_llm_evaluation_completeness = Gauge('rag_llm_evaluation_completeness', 'LLM evaluation completeness score', registry=registry)
rag_llm_evaluation_clarity = Gauge('rag_llm_evaluation_clarity', 'LLM evaluation clarity score', registry=registry)
rag_llm_evaluation_faithfulness = Gauge('rag_llm_evaluation_faithfulness', 'LLM evaluation faithfulness score', registry=registry)

# User Feedback Metrics
rag_user_feedback_positive = Counter('rag_user_feedback_positive_total', 'Total positive user feedback', registry=registry)
rag_user_feedback_negative = Counter('rag_user_feedback_negative_total', 'Total negative user feedback', registry=registry)
rag_user_satisfaction_rate = Gauge('rag_user_satisfaction_rate', 'User satisfaction rate (positive feedback ratio)', registry=registry)

# System Health Metrics
qdrant_collection_vectors_count = Gauge('qdrant_collection_vectors_count', 'Number of vectors in collection', ['collection'], registry=registry)
ollama_model_loaded = Gauge('ollama_model_loaded', 'Whether model is loaded', ['model'], registry=registry)

app = FastAPI(title="RAG Metrics Exporter")

class MetricsCollector:
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        
    def collect_qdrant_metrics(self):
        """Collect metrics from Qdrant"""
        try:
            # Get collection info
            response = requests.get(f"{self.qdrant_url}/collections/godot-docs", timeout=5)
            if response.status_code == 200:
                data = response.json()
                points_count = data.get("result", {}).get("points_count", 0)
                qdrant_collection_vectors_count.labels(collection="godot-docs").set(points_count)
                logger.info(f"Qdrant collection has {points_count} vectors")
        except Exception as e:
            logger.error(f"Failed to collect Qdrant metrics: {e}")
    
    def collect_ollama_metrics(self):
        """Collect metrics from Ollama"""
        try:
            # Check if model is loaded
            response = requests.get(f"{self.ollama_url}/api/ps", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                for model in models:
                    model_name = model.get("name", "unknown")
                    ollama_model_loaded.labels(model=model_name).set(1)
                    logger.info(f"Ollama model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to collect Ollama metrics: {e}")

# Initialize collector
collector = MetricsCollector()

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    # Collect fresh metrics
    collector.collect_qdrant_metrics()
    collector.collect_ollama_metrics()
    
    return PlainTextResponse(
        generate_latest(registry),
        media_type="text/plain"
    )

@app.post("/metrics/query")
async def record_query_metrics(request: Request):
    """Record metrics for a query"""
    try:
        data = await request.json()
        
        # Record query
        rag_queries_total.inc()
        
        # Record response time if provided
        if "response_time" in data:
            rag_response_time_seconds.observe(data["response_time"])
        
        # Record query type
        query_type = classify_query(data.get("query", ""))
        rag_query_types_total.labels(query_type=query_type).inc()
        
        # Record vector search
        rag_vector_searches_total.inc()
        
        # Record LLM evaluation scores if provided
        if "evaluation_scores" in data:
            scores = data["evaluation_scores"]
            if "relevance" in scores:
                rag_llm_evaluation_relevance.set(scores["relevance"])
            if "accuracy" in scores:
                rag_llm_evaluation_accuracy.set(scores["accuracy"])
            if "completeness" in scores:
                rag_llm_evaluation_completeness.set(scores["completeness"])
            if "clarity" in scores:
                rag_llm_evaluation_clarity.set(scores["clarity"])
            if "faithfulness" in scores:
                rag_llm_evaluation_faithfulness.set(scores["faithfulness"])
        
        # Record user feedback if provided
        if "feedback" in data:
            feedback = data["feedback"]
            if feedback == "positive":
                rag_user_feedback_positive.inc()
            elif feedback == "negative":
                rag_user_feedback_negative.inc()
            
            # Update satisfaction rate
            positive_count = rag_user_feedback_positive._value._value
            negative_count = rag_user_feedback_negative._value._value
            total_feedback = positive_count + negative_count
            
            if total_feedback > 0:
                satisfaction_rate = positive_count / total_feedback
                rag_user_satisfaction_rate.set(satisfaction_rate)
        
        return {"status": "recorded"}
    
    except Exception as e:
        logger.error(f"Failed to record metrics: {e}")
        return {"status": "error", "message": str(e)}

def classify_query(query: str) -> str:
    """Classify query into categories"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["collision", "physics", "rigidbody", "characterbody"]):
        return "physics"
    elif any(word in query_lower for word in ["camera", "view", "viewport"]):
        return "camera"
    elif any(word in query_lower for word in ["scene", "node", "tree"]):
        return "scene_management"
    elif any(word in query_lower for word in ["input", "movement", "player", "control"]):
        return "input_control"
    elif any(word in query_lower for word in ["ui", "menu", "button", "interface"]):
        return "ui"
    elif any(word in query_lower for word in ["script", "code", "function", "class"]):
        return "scripting"
    else:
        return "general"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    logger.info("Starting RAG Metrics Exporter")
    uvicorn.run(app, host="0.0.0.0", port=8000)
