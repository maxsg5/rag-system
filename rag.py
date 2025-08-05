from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText

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
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "filter": None})

# Setup Ollama LLM
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2:1b",
    temperature=0.2
)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a Godot Engine expert AI assistant helping users understand game development concepts.
Use ONLY the provided context to answer the user's question.
If the answer cannot be found in the context, say you don’t know — do not make up an answer.

Instructions:
- Be concise and technically accurate.
- Include code examples or steps if relevant.
- Prefer referencing node types, menus, and actual terms used in the docs.

Context:
{context}

Question:
{question}

Answer:
""".strip()
)

# Build RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Query
while True:
    query = input("🔎 Ask a question (or 'exit'): ")
    if query.strip().lower() == "exit":
        break

    # Optional: generate a metadata keyword for filtering
    keyword = query.lower().split()[0]  # crude example: first word
    hybrid_filter = Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchText(text=keyword)
            )
        ]
    )

    retriever.search_kwargs["filter"] = hybrid_filter

    result = rag_chain.invoke({"query": query})
    answer = result["result"]
    sources = result["source_documents"]

    print("\n🧠 Answer:\n" + "-" * 80)
    print(answer.strip())
    
    print("\n📚 Sources:")
    unique_sources = set(doc.metadata.get("source", "unknown") for doc in sources)
    for src in unique_sources:
        print(f"- {src}")
