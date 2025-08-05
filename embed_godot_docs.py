from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain_core.documents import Document
from godot_docs_processor import GodotDocsProcessor
import uuid

processor = GodotDocsProcessor()

# TODO: COMMENT THIS OUT AFTER FIRST RUN
# processor.prepare()


# Load chunked docs
docs = processor.load_documents("chunked/chunked_docs.json")

# Embed model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Create collection (if not exists)
collection_name = "rag_docs"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=qmodels.VectorParams(
        size=embedding_model.embedding_size,
        distance=qmodels.Distance.COSINE
    )
)

# Upload
qdrant = Qdrant.from_documents(
    documents=docs,
    embedding=embedding_model,
    client=client,
    collection_name=collection_name
)
