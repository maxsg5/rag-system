import streamlit as st
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from qdrant_client.models import Filter, FieldCondition, MatchText

# -- Init model and retriever
@st.cache_resource
def setup_rag_chain():
    # Embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",  # or any fast model
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Qdrant
    vectorstore = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="rag_docs",
        embeddings=embedding
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Prompt
    prompt = PromptTemplate(
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

    # LLM (Ollama)
    llm = ChatOllama(
        model="llama3.2:1b",
        base_url="http://localhost:11434",
        temperature=0.2
    )

    # RAG Chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

rag_chain = setup_rag_chain()

# -- UI
st.set_page_config(page_title="Godot Docs Assistant", layout="wide")
st.title("🤖 Godot Docs Assistant (RAG + Ollama)")

query = st.text_input("Ask a question about Godot documentation:", placeholder="e.g. how to add a camera?")

if query:
    with st.spinner("Thinking..."):
        result = rag_chain.invoke({"query": query})
        answer = result["result"]
        sources = result["source_documents"]

    st.markdown("### 🧠 Answer")
    st.markdown(answer)

    st.markdown("---")
    st.markdown("### 📚 Source Documents")
    for i, doc in enumerate(sources):
        with st.expander(f"🔹 Source {i+1}: {doc.metadata.get('source', 'unknown')}"):
            st.code(doc.page_content.strip()[:1500])
