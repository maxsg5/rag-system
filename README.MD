# Steps to Reproduce

1. Start the Docker containers.
```bash
docker compose up -d
```

# Cosine Similarity
Range: 0.0 to 1.0 (higher = more relevant)
Model: all-MiniLM-L6-v2 (384-dimensional embeddings)
Distance Metric: Cosine distance in vector space
Process:
Query Embedding: Your question → 384D vector
Document Embeddings: Pre-stored 384D vectors for all Godot docs
Similarity Search: Cosine similarity between query and all documents
Ranking: Top 5 most similar documents returned
Score Interpretation:
0.7-1.0: High relevance (excellent matches)
0.4-0.7: Medium relevance (related concepts)
<0.4: Low relevance (weakly related)
