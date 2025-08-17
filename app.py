import streamlit as st
st.set_page_config(page_title="Godot Docs Assistant", layout="wide")

import os
import time
import requests
import json
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

# Metrics tracking function
def send_metrics(query: str, response_time: float, evaluation_scores: dict = None, feedback: str = None):
    """Send metrics to monitoring system"""
    try:
        metrics_url = os.getenv("METRICS_URL", "http://localhost:8000/metrics/query")
        data = {
            "query": query,
            "response_time": response_time,
            "timestamp": time.time()
        }
        if evaluation_scores:
            data["evaluation_scores"] = evaluation_scores
        if feedback:
            data["feedback"] = feedback
        
        requests.post(metrics_url, json=data, timeout=5)
    except Exception as e:
        # Don't let metrics failures break the app
        pass

# LLM-as-a-Judge Evaluation Functions
def evaluate_answer_quality(query: str, answer: str, context: str, llm) -> dict:
    """Evaluate answer quality using LLM-as-a-judge approach"""
    
    evaluation_prompt = f"""
You are an expert evaluator assessing the quality of AI-generated answers about Godot game engine documentation.

Evaluate the following answer on these criteria (score 1-5 for each):

**RELEVANCE**: How well does the answer address the specific question?
**ACCURACY**: Is the technical information correct based on the provided context?
**COMPLETENESS**: Does the answer provide sufficient detail to be helpful?
**CLARITY**: Is the answer clear and well-structured?
**FAITHFULNESS**: Does the answer stay true to the provided context without hallucinating?

**Question**: {query}

**Context**: {context}

**Answer**: {answer}

**EVALUATION FORMAT**:
RELEVANCE: [score]/5 - [brief explanation]
ACCURACY: [score]/5 - [brief explanation]
COMPLETENESS: [score]/5 - [brief explanation]
CLARITY: [score]/5 - [brief explanation]
FAITHFULNESS: [score]/5 - [brief explanation]

OVERALL: [average score]/5
SUMMARY: [2-3 sentence overall assessment]
"""
    
    try:
        evaluation = llm.invoke(evaluation_prompt).content
        return {"evaluation": evaluation, "prompt": evaluation_prompt}
    except Exception as e:
        return {"evaluation": f"Evaluation failed: {str(e)}", "prompt": evaluation_prompt}

def parse_evaluation_scores(evaluation_text: str) -> dict:
    """Extract numeric scores from evaluation text"""
    scores = {}
    lines = evaluation_text.split('\n')
    
    criteria = ['RELEVANCE', 'ACCURACY', 'COMPLETENESS', 'CLARITY', 'FAITHFULNESS', 'OVERALL']
    
    for line in lines:
        for criterion in criteria:
            if line.strip().startswith(criterion):
                try:
                    # Extract score (look for pattern like "4/5" or "4.2/5")
                    score_part = line.split(':')[1].strip()
                    score = float(score_part.split('/')[0].strip())
                    scores[criterion.lower()] = score
                except:
                    pass
    
    return scores

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
st.caption(
    f"Collection: {os.getenv('QDRANT_COLLECTION', 'godot-docs')} | Embeddings: {os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')} | Ollama: {os.getenv('OLLAMA_MODEL', 'llama3.2:1b')}"
)

with st.expander("ℹ️ How the Relevance Scoring Works"):
    st.markdown("""
    **Vector Similarity Search Process:**
    
    1. **Query Processing**: Your question is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` embedding model
    
    2. **Document Embeddings**: All Godot documentation chunks were pre-processed and embedded using the same model during indexing
    
    3. **Similarity Calculation**: Qdrant uses **cosine similarity** to compare your query vector with all document vectors:
       - Formula: `cosine_similarity = dot(query_vector, doc_vector) / (||query_vector|| * ||doc_vector||)`
       - This measures the angle between vectors in 384-dimensional space
    
    4. **Distance to Similarity**: Qdrant stores cosine distance (1 - cosine_similarity), then converts back to similarity for scoring
    
    5. **Score Range**: 
       - **0.0**: No semantic similarity (perpendicular vectors)
       - **1.0**: Perfect semantic match (identical vectors)
       - **0.7-1.0**: High relevance (typical for good matches)
       - **0.4-0.7**: Medium relevance (related concepts)
       - **<0.4**: Low relevance (weakly related)
    
    The top 5 most similar documents are retrieved and used as context for the LLM to generate the answer.
    """)

query = st.text_input("Ask a question about Godot documentation:", placeholder="e.g. how to add a camera?")

if query:
    # Create unique keys for this specific query
    query_hash = hash(query)
    answer_key = f"answer_{query_hash}"
    sources_key = f"sources_{query_hash}"
    feedback_key = f"feedback_{query_hash}"
    
    # Initialize feedback state for this query
    if feedback_key not in st.session_state:
        st.session_state[feedback_key] = {"given": False, "type": None}
    
    # Only run the query if we don't have cached results
    if answer_key not in st.session_state:
        start_time = time.time()
        
        with st.spinner("Thinking..."):
            result = rag_chain.invoke({"query": query})
            answer = result["result"]
            sources = result["source_documents"]
        
        response_time = time.time() - start_time
        
        # Run automatic LLM-as-a-Judge evaluation
        with st.spinner("Running quality evaluation..."):
            try:
                # Combine source documents into context
                context = "\n\n".join([
                    f"Source {i+1}: {doc.page_content[:500]}..."
                    for i, doc in enumerate(sources)
                ])
                
                # Get the same LLM instance for evaluation
                llm = ChatOllama(
                    model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    temperature=0.1  # Lower temperature for more consistent evaluation
                )
                
                # Run evaluation
                eval_result = evaluate_answer_quality(query, answer, context, llm)
                scores = parse_evaluation_scores(eval_result["evaluation"])
                
                # Cache evaluation results
                st.session_state[f"eval_{query_hash}"] = eval_result
                st.session_state[f"scores_{query_hash}"] = scores
                
                # Send metrics with evaluation scores
                send_metrics(query, response_time, scores)
            except Exception as e:
                # If evaluation fails, still send basic metrics
                send_metrics(query, response_time)
                st.session_state[f"eval_{query_hash}"] = {"evaluation": f"Evaluation failed: {str(e)}", "prompt": ""}
                st.session_state[f"scores_{query_hash}"] = {}
        
        # Cache the results so they don't change on rerun
        st.session_state[answer_key] = answer
        st.session_state[sources_key] = sources
        st.session_state[f"response_time_{query_hash}"] = response_time
    else:
        # Use cached results
        answer = st.session_state[answer_key]
        sources = st.session_state[sources_key]
        response_time = st.session_state[f"response_time_{query_hash}"]

    st.markdown("### 🧠 Answer")
    st.markdown(answer)
    
    # User Feedback Section
    st.markdown("---")
    st.markdown("### 👍 Rate this answer")
    
    current_feedback = st.session_state[feedback_key]
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("👍 Helpful", 
                    key=f"thumbs_up_{query_hash}", 
                    disabled=current_feedback["given"]):
            send_metrics(query, response_time, feedback="positive")
            st.session_state[feedback_key] = {"given": True, "type": "positive"}
    
    with col2:
        if st.button("👎 Not Helpful", 
                    key=f"thumbs_down_{query_hash}", 
                    disabled=current_feedback["given"]):
            send_metrics(query, response_time, feedback="negative")
            st.session_state[feedback_key] = {"given": True, "type": "negative"}
    
    with col3:
        if st.button("🔄 Reset Feedback", key=f"clear_feedback_{query_hash}"):
            st.session_state[feedback_key] = {"given": False, "type": None}
    
    # Show feedback status
    if current_feedback["given"]:
        if current_feedback["type"] == "positive":
            st.success("✅ Thanks for your positive feedback! 👍")
        elif current_feedback["type"] == "negative":
            st.error("📝 Thanks for your feedback. We'll work to improve! 👎")
    
    # LLM-as-a-Judge Evaluation Results (Automatic)
    st.markdown("---")
    st.markdown("### 🏆 Quality Evaluation (Automatic)")
    
    # Get cached evaluation results
    eval_result = st.session_state.get(f"eval_{query_hash}", {})
    scores = st.session_state.get(f"scores_{query_hash}", {})
    
    if scores:
        st.markdown("#### 📊 Quality Scores")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'relevance' in scores:
                st.metric("🎯 Relevance", f"{scores['relevance']:.1f}/5")
            if 'accuracy' in scores:
                st.metric("✅ Accuracy", f"{scores['accuracy']:.1f}/5")
        
        with col2:
            if 'completeness' in scores:
                st.metric("📋 Completeness", f"{scores['completeness']:.1f}/5")
            if 'clarity' in scores:
                st.metric("💡 Clarity", f"{scores['clarity']:.1f}/5")
        
        with col3:
            if 'faithfulness' in scores:
                st.metric("🔒 Faithfulness", f"{scores['faithfulness']:.1f}/5")
            if 'overall' in scores:
                st.metric("🏆 Overall", f"{scores['overall']:.1f}/5", delta=None)
    
    # Detailed evaluation in expandable section
    if eval_result.get("evaluation"):
        with st.expander("📝 View Detailed Evaluation"):
            st.markdown("#### Full LLM-as-a-Judge Assessment")
            st.text_area("Evaluation Details", eval_result["evaluation"], height=300, key=f"eval_text_{query_hash}")
            
            if eval_result.get("prompt"):
                with st.expander("🔧 Evaluation Prompt Used"):
                    st.code(eval_result["prompt"], language="text")
    elif eval_result:
        st.warning("⚠️ Quality evaluation was attempted but encountered an issue. Basic metrics were still recorded.")
    
    # Evaluation criteria explanation
    with st.expander("📚 Quality Evaluation Criteria"):
        st.markdown("""
        **Automatic LLM-as-a-Judge Evaluation:**
        
        Every answer is automatically evaluated across 5 dimensions:
        
        - **🎯 Relevance**: How well does the answer address the specific question asked?
        - **✅ Accuracy**: Is the technical information correct based on Godot documentation?
        - **📋 Completeness**: Does the answer provide sufficient detail to be actionable?
        - **💡 Clarity**: Is the answer well-structured and easy to understand?
        - **🔒 Faithfulness**: Does the answer stay true to the source context without adding false information?
        
        **Scoring Scale**: 1 (Poor) → 2 (Fair) → 3 (Good) → 4 (Very Good) → 5 (Excellent)
        
        These scores are automatically sent to the monitoring system and displayed in the Grafana dashboard.
        """)

    st.markdown("---")
    st.markdown("### 📚 Source Documents")
    st.info("""
    **Relevance Score Calculation:**
    - Scores are calculated using **cosine similarity** between your query embedding and document embeddings
    - Range: 0.0 to 1.0 (higher = more relevant)
    - Method: Your query is converted to a 384-dimensional vector using the all-MiniLM-L6-v2 model
    - Each document was pre-embedded using the same model during indexing
    - Qdrant performs vector search using cosine distance to find the most semantically similar documents
    """)
    
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

# Add monitoring links to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Monitoring")
st.sidebar.markdown("[📈 Grafana Dashboard](http://localhost:3000/d/rag_dashboard)")
st.sidebar.markdown("[🔍 Prometheus](http://localhost:9090)")
st.sidebar.markdown("[📊 Raw Metrics](http://localhost:8000/metrics)")
