from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from godot_docs_processor import GodotDocsProcessor
import uuid
import os

processor = GodotDocsProcessor()

# Load chunked docs
print("📄 Loading chunked documents...")
docs = processor.load_documents("data/chunked/chunked_docs.json")
print(f"✅ Loaded {len(docs)} documents")

# Embed model
print("🤖 Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'},  # Specify device explicitly
    encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings
)
print("✅ Embedding model loaded")

# Connect to Qdrant
print("🔗 Connecting to Qdrant...")
client = QdrantClient(host="localhost", port=6333)

# Create collection (if not exists)
collection_name = "rag_docs"

# Check if collection exists, if not create it
try:
    collection_info = client.get_collection(collection_name)
    print(f"Collection '{collection_name}' already exists with {collection_info.points_count} points")
    
    # Ask user if they want to recreate
    user_input = input("Do you want to recreate the collection? (y/N): ").strip().lower()
    if user_input in ['y', 'yes']:
        print(f"🗑️ Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)
        
        print(f"🆕 Creating new collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=384, # Size of the embedding vector
                distance=qmodels.Distance.COSINE
            )
        )
    else:
        print("Using existing collection...")
        qdrant = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embedding_model
        )
        print("✅ Qdrant ingestion complete (using existing collection).")
        exit(0)
        
except Exception:
    print(f"Collection '{collection_name}' does not exist, creating new one...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=384, # Size of the embedding vector
            distance=qmodels.Distance.COSINE
        )
    )

print(f"📤 Uploading {len(docs)} documents to Qdrant (this may take a while)...")
# Upload documents to Qdrant with progress feedback
qdrant = Qdrant.from_documents(
    documents=docs,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name=collection_name,
    batch_size=100  # Process in smaller batches
)


print("✅ Qdrant ingestion complete.")
