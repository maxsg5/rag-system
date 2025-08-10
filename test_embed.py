import os
import json
import uuid
import torch
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings  # new package per deprecation notice
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Load your chunks
with open("data/chunked/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Initialize embedding model on GPU
print("🤖 Loading embedding model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # Or try: 'BAAI/bge-small-en-v1.5', 'thenlper/gte-small'
    model_kwargs={'device': device},  # Use GPU if available
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Embedding model loaded.")

# Connect to Qdrant
COLLECTION_NAME = "godot-docs"
qdrant = QdrantClient("http://localhost:6333")

# Ensure collection exists; if already present ask user before overwriting
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 is 384-dim
if qdrant.collection_exists(COLLECTION_NAME):
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        existing_points = getattr(info, 'points_count', None) or getattr(info, 'vectors_count', None)
    except Exception:
        existing_points = None
    prompt = f"Collection '{COLLECTION_NAME}' already exists"
    if existing_points is not None:
        prompt += f" with ~{existing_points} points"
    prompt += ". Overwrite (recreate)? [y/N]: "
    choice = input(prompt).strip().lower()
    if choice in ("y", "yes"):
        print("🗑️ Deleting existing collection...")
        qdrant.delete_collection(COLLECTION_NAME)
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print("✅ Collection recreated.")
    else:
        print("➡️ Reusing existing collection (points will be added; may duplicate if already embedded).")
else:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print("✅ Collection created.")

# Embed + upsert in batches
BATCH_SIZE = 64
batch = []

for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks")):
    try:
        vec = embedding_model.embed_query(chunk["text"])
        # Keep the text so we can display it later; optionally truncate to save space
        metadata = dict(chunk)
        # Optionally: metadata["text"] = chunk["text"][:2000]

        point = PointStruct(
            id=chunk.get("id", str(uuid.uuid4())),
            vector=vec,
            payload=metadata
        )
        batch.append(point)

        if len(batch) >= BATCH_SIZE or i == len(chunks) - 1:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
            batch = []

    except Exception as e:
        print(f"❌ Error on chunk {i}: {e}")


query = "How do I detect collisions in 2D?"
vec = embedding_model.embed_query(query)

results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=vec,  # direct vector query (list of floats)
    limit=5,
    with_payload=True
)

# Support multiple possible return shapes across qdrant-client versions
if hasattr(results, 'points'):
    iter_points = results.points  # QueryResponse style
else:
    iter_points = results  # already a list/iterable

for raw_hit in iter_points:
    # Some versions may return tuples like (ScoredPoint, extra)
    hit = raw_hit[0] if isinstance(raw_hit, tuple) and raw_hit else raw_hit
    payload = getattr(hit, 'payload', {}) or {}
    score = getattr(hit, 'score', None)
    text = payload.get("text") or payload.get("content") or "<no text stored>"
    print("Title:", payload.get("title"))
    print("Score:", score)
    print("Chunk:", text[:300] if isinstance(text, str) else str(text)[:300], "\n---\n")
