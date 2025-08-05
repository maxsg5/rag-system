from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient

# Setup embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to Qdrant
qdrant = QdrantClient(host="localhost", port=6333)
collection_name = "rag_docs"

# Load retriever from Qdrant
vectorstore = Qdrant(
    client=qdrant,
    collection_name=collection_name,
    embeddings=embedding
)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# Setup Ollama LLM
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2:1b",
    temperature=0.2
)

# Build RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Query
while True:
    query = input("🔎 Ask a question (or 'exit'): ")
    if query.strip().lower() == "exit":
        break

    result = rag_chain.invoke({"query": query})

    print("\n🧠 Answer:")
    print(result["result"])

    print("\n📚 Sources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source', 'unknown')}")
