# Godot RAG System
This project was implemented for [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) - a free course about LLMs and RAG.

## Problem Description

**The Challenge:** Godot is a powerful open-source game engine, but its extensive documentation can be overwhelming for newcomers. Developers, especially those new to game engines, often struggle to find relevant information quickly when they encounter specific problems or need to implement particular features. Traditional documentation search is often inadequate - users may not know the exact terminology to search for, or they might need information scattered across multiple sections.

**The Solution:** This RAG (Retrieval-Augmented Generation) system transforms how developers interact with Godot documentation. Instead of manually searching through hundreds of pages, users can ask natural language questions and receive contextual answers along with direct links to the most relevant documentation sections. The system combines semantic search with AI-powered response generation to provide accurate, helpful answers that accelerate learning and development workflow.

**Key Benefits:**
- **Faster Problem Resolution:** Get instant answers without browsing through extensive documentation
- **Contextual Learning:** Receive targeted information relevant to your specific use case
- **Documentation Discovery:** Find relevant sections you might not have discovered through traditional search
- **Beginner-Friendly:** Ask questions in plain English without needing to know exact technical terminology

# Dataset

The dataset used in this project contains comprehensive information from the official Godot Engine documentation, including:

**Document Structure**: Each chunk contains structured information from Godot's documentation hierarchy
- **Title**: The main title of the documentation page (e.g., "RigidBody2D", "Creating Your First Scene")
- **File Path**: The source documentation file location (e.g., "classes/class_rigidbody2d.rst.txt")
- **Section**: The major section within the documentation (e.g., "Methods", "Properties", "Tutorials")
- **Subsection**: Specific subsections for detailed organization (e.g., "Virtual Methods", "Constants")
- **Content**: The actual documentation text containing explanations, code examples, and instructions

**Content Categories**: The documentation covers all aspects of Godot game development:
- **Class References**: Detailed API documentation for all Godot classes and methods
- **Tutorials**: Step-by-step guides for common game development tasks
- **Manual Pages**: Comprehensive explanations of Godot's features and concepts
- **Code Examples**: GDScript code snippets with explanations
- **Best Practices**: Recommended approaches and patterns for game development

**Processing Pipeline**:
1. **Download**: Official Godot documentation is downloaded from the nightly builds
2. **Extraction**: Documentation is extracted and converted from .rst.txt files to structured text
3. **Chunking**: Large documents are intelligently split into smaller, semantically meaningful chunks
4. **Embedding**: Each chunk is converted into 384-dimensional vectors using the all-MiniLM-L6-v2 model
5. **Storage**: Vectors and metadata are stored in Qdrant vector database for fast semantic search

The processed dataset contains thousands of documentation chunks optimized for retrieval-augmented generation. Each chunk maintains its connection to the original Godot documentation through metadata, enabling users to trace answers back to their official sources.

You can find the processed data in `data/chunked/chunks.json` and the raw documentation can be found at https://nightly.link/godotengine/godot-docs/workflows/build_offline_docs/master/godot-docs-epub-stable.zip.


# Project Demo Video
https://www.youtube.com/embed/-F6iT-kqeKw?si=ReaCcVhH0xVhkLIo

# Screenshots
![Screenshot 1](images/Screenshot%202025-09-07%20181037.png)
![Screenshot 2](images/Screenshot%202025-09-07%20181210.png)
![Screenshot 3](images/Screenshot%202025-09-07%20181240.png)
![Screenshot 4](images/Screenshot%202025-09-07%20181324.png)


# Prerequisites
Have the following packages and tools installed:

- Python 3.12 or later
- Docker
- Docker Compose

optional but recommended

- 32GB+ RAM
- NVIDIA GPU (for faster embedding computation, and LLM otherwise CPU will be used)

# How to run
## You have two options.

### Option 1 (*pre-chunked*, saves you more time, uses dataset that was pre-chunked on 8/17/2025) - recommended

```bash
./setup_pre_chunked.sh
```

### Option 2 (*from scratch*, downloads latest docs and chunks the data)

```bash
./setup_scratch.sh
```


# Monitoring
1. **Query Rate (QPS)** - Real-time query volume
2. **Average Response Time** - Response latency gauge  
3. **LLM Evaluation Scores** - AI quality metrics (Relevance, Accuracy, Completeness, Clarity, Faithfulness)
4. **Vector Database Metrics** - Qdrant performance and usage
5. **Top Query Categories** - Most common query types

## Service URLs

- 📊 **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- 🔍 **Prometheus**: http://localhost:9090
- 📈 **Metrics**: http://localhost:8000/metrics


# Evaluation
## Cosine Similarity
### Range: 0.0 to 1.0 (higher = more relevant)

#### Model: 
- all-MiniLM-L6-v2 (384-dimensional embeddings)

#### Distance Metric: 
- Cosine distance in vector space
Process:
Query Embedding: Your question → 384D vector

#### Document Embeddings: 
- Pre-stored 384D vectors for all Godot docs

#### Similarity Search: 
- Cosine similarity between query and all documents
#### Ranking:
- Top 5 most similar documents returned

### Score Interpretation:
- 0.7-1.0: High relevance (excellent matches)
- 0.4-0.7: Medium relevance (related concepts)
- <0.4: Low relevance (weakly related)

## LLM as a Judge

The LLM (Large Language Model) is used as a judge to evaluate the quality of the answers generated by the RAG system. This involves using the LLM to assess various aspects of the answers, such as relevance, accuracy, completeness, clarity, and faithfulness.


# Packages and Tools

## Core Infrastructure
- **Docker & Docker Compose**: Container orchestration for services
- **Python 3.12+**: Main programming language
- **Bash Scripts**: Automated setup and deployment scripts

## RAG System Components

### **Vector Database & Embeddings**
- **`qdrant-client`**: Vector database client for semantic search
- **`sentence-transformers`**: Pre-trained embedding models
- **`langchain_huggingface`**: HuggingFace integration for embeddings
- **`transformers`**: Deep learning models for NLP
- **`torch`**: PyTorch for GPU-accelerated embedding computation

### **Language Model & Chain**
- **`langchain`**: RAG pipeline framework
- **`langchain-core`**: Core LangChain functionality
- **`langchain-community`**: Community integrations
- **`langchain_qdrant`**: Qdrant vector store integration
- **`langchain_ollama`**: Ollama LLM integration
- **`Ollama`**: Local LLM inference server (llama3.2:1b model)
- **`tiktoken`**: Token counting and text processing

### **Web Interface**
- **`streamlit`**: Interactive web application framework

### **Data Processing**
- **`requests`**: HTTP client for downloading documentation
- **`bs4` (BeautifulSoup)**: parsing and web scraping
- **`pillow`**: Image processing support (note: future enhancement, not implemented in the pipeline)
- **`tqdm`**: Progress bars for data processing

## Monitoring & Metrics Stack

### **Metrics Collection**
- **`prometheus_client`**: Python client for Prometheus metrics
- **`fastapi`**: High-performance API framework for metrics endpoint
- **`uvicorn`**: ASGI server for FastAPI applications
- **`python-multipart`**: Multipart form data support

### **Monitoring Services (Docker)**
- **Prometheus**: Time-series metrics collection and storage
- **Grafana**: Metrics visualization and dashboards
- **Qdrant**: High-performance vector database

## Development & Deployment Tools
- **Shell Scripts**: 
  - `setup_pre_chunked.sh`: Quick setup with pre-processed data
  - `setup_scratch.sh`: Full pipeline from raw documentation
  - `start_monitoring.sh`: Launch monitoring stack
  - `import_dashboard.sh`: Configure Grafana dashboards

## Key Models & Algorithms
- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional vectors)
- **LLM**: `llama3.2:1b` (1 billion parameter model)
- **Similarity Search**: Cosine similarity in vector space
- **Evaluation**: LLM-as-a-Judge for answer quality assessment


## Troubleshooting

### Grafana Dashboard Issues

If the dashboard doesn't appear automatically:
1. Make sure you've logged into Grafana web interface first (admin/admin)
2. Run `./import_dashboard.sh` manually
3. Check that all services are running with `docker compose ps`
4. Verify Grafana is accessible at http://localhost:3000



# Future Considerations
- Compare different LLM models (e.g. Llama 2, Falcon, Mistral)
- Compare different chunking methods (e.g. overlapping chunks, hierarchical chunking)
- Compare different databases (e.g. Pinecone, Weaviate, Milvus)
- Compare different retrieval techniques (e.g. Maximal Marginal Relevance)
- Explore additional data sources for improving knowledge base (e.g. Wikipedia, GitHub, Stack Overflow)
- Use user feedback for continuous improvement and fine-tuning.
- Deploy on cloud platforms for scalability and availability.
