import streamlit as st
st.set_page_config(page_title="Godot Docs Assistant", layout="wide")

import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from typing import List
from qdrant_client.models import Filter, FieldCondition, MatchText

# Custom Qdrant Retriever that preserves metadata
class QdrantRetriever(BaseRetriever):
    client: QdrantClient
    collection_name: str
    embedding_model: HuggingFaceEmbeddings
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, client: QdrantClient, collection_name: str, embedding_model, k: int = 5):
        super().__init__(client=client, collection_name=collection_name, embedding_model=embedding_model, k=k)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Get embedding for the query
        query_vector = self.embedding_model.embed_query(query)
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=self.k,
            with_payload=True
        )
        
        # Convert to LangChain Documents with proper metadata
        documents = []
        for result in results:
            # Extract text from payload
            text = result.payload.get('text', '')
            
            # Create metadata dict excluding 'text' field
            metadata = {k: v for k, v in result.payload.items() if k != 'text'}
            metadata['score'] = result.score  # Add relevance score
            
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        return documents

# -- Init model and retriever
@st.cache_resource
def setup_rag_chain():
    # Embeddings
    # IMPORTANT: Must match the model used to build the stored vectors.
    # Your indexing script used all-MiniLM-L6-v2 (384-dim). We'll default to that.
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Qdrant - Use custom retriever to properly handle metadata
    collection_name = os.getenv("QDRANT_COLLECTION", "godot-docs")
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    
    retriever = QdrantRetriever(
        client=client,
        collection_name=collection_name,
        embedding_model=embedding,
        k=5
    )

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
st.title("🤖 Godot Docs Assistant (RAG + Ollama)")
st.caption(f"Using collection: {os.getenv('QDRANT_COLLECTION', 'godot-docs')} | model: {os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')}")

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
        # Extract meaningful title from metadata
        title = doc.metadata.get('title', 'Unknown Title')
        file_path = doc.metadata.get('file_path', 'Unknown File')
        section = doc.metadata.get('section', '')
        subsection = doc.metadata.get('subsection', '')
        score = doc.metadata.get('score', 'N/A')
        
        # Build a more informative title
        if section and section != file_path:
            display_title = f"{title} → {section}"
        else:
            display_title = f"{title} ({file_path})"
            
        if subsection and subsection != 'None':
            display_title += f" → {subsection}"
        
        # Generate Godot docs URL
        # Convert file_path from .rst.txt to proper web URL
        if file_path and file_path != 'Unknown File':
            # Remove .rst.txt extension and convert to web path
            web_path = file_path.replace('.rst.txt', '.html')
            godot_docs_url = f"https://docs.godotengine.org/en/stable/{web_path}"
            
            # Create clickable title with link
            title_with_link = f"[{display_title}]({godot_docs_url})"
        else:
            title_with_link = display_title
        
        with st.expander(f"🔹 Source {i+1}: {display_title}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**File:** `{file_path}`")
                if file_path and file_path != 'Unknown File':
                    st.markdown(f"**📖 View in Godot Docs:** {title_with_link}")
                if section: st.write(f"**Section:** {section}")
                if subsection and subsection != 'None': st.write(f"**Subsection:** {subsection}")
            with col2:
                st.metric("Relevance Score", f"{score:.3f}" if isinstance(score, (int, float)) else score)
            
            st.write("**Content:**")
            st.code(doc.page_content.strip()[:1500])
