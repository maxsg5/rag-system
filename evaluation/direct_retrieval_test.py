"""
Direct Qdrant retrieval evaluation using the actual data structure.
Bypasses LangChain to test retrieval methods directly against your Qdrant database.
"""

import os
import time
import numpy as np
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.models import SearchRequest, Filter, FieldCondition, MatchText


class DirectRetrievalTester:
    """Direct retrieval testing using Qdrant client without LangChain wrapper."""
    
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "godot-docs"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Test queries - real Godot questions
        self.test_queries = [
            "how to move a 2D character with input",
            "add a camera to follow the player", 
            "create an animation in Godot",
            "collision detection between two bodies",
            "how to export a game",
            "add sound effects to a scene",
            "create a UI button",
            "load a different scene",
            "save and load game data",
            "use signals in Godot"
        ]
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query using the same model used for indexing."""
        return self.embedding_model.embed_query(query)
    
    def similarity_search(self, query: str, k: int = 5) -> Tuple[List[Dict], float]:
        """Direct similarity search using Qdrant."""
        start_time = time.time()
        
        # Embed the query
        query_vector = self.embed_query(query)
        
        # Search Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True
        )
        
        search_time = (time.time() - start_time) * 1000
        
        # Format results
        results = []
        for point in search_result:
            results.append({
                'id': str(point.id),
                'content': point.payload.get('text', ''),
                'title': point.payload.get('title', ''),
                'file_path': point.payload.get('file_path', ''),
                'section': point.payload.get('section', ''),
                'subsection': point.payload.get('subsection', ''),
                'score': float(point.score),
                'method': 'Similarity'
            })
        
        return results, search_time
    
    def mmr_search(self, query: str, k: int = 5, lambda_mult: float = 0.5) -> Tuple[List[Dict], float]:
        """
        MMR (Maximum Marginal Relevance) search implementation.
        Balances relevance with diversity to reduce redundancy.
        """
        start_time = time.time()
        
        # First get more candidates than needed for MMR selection
        candidates_k = min(k * 3, 20)  # Get 3x candidates or max 20
        
        # Embed the query
        query_vector = self.embed_query(query)
        
        # Get initial candidates
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=candidates_k,
            with_payload=True
        )
        
        if not search_result:
            return [], (time.time() - start_time) * 1000
        
        # Convert to list for MMR processing
        candidates = []
        candidate_vectors = []
        
        for point in search_result:
            candidates.append({
                'id': str(point.id),
                'content': point.payload.get('text', ''),
                'title': point.payload.get('title', ''),
                'file_path': point.payload.get('file_path', ''),
                'section': point.payload.get('section', ''),
                'subsection': point.payload.get('subsection', ''),
                'score': float(point.score),
                'method': 'MMR'
            })
            # Note: In a real implementation, we'd need the document vectors
            # For this demo, we'll use a simplified MMR that just ensures diversity
        
        # Simplified MMR: select diverse results based on content similarity
        selected = []
        remaining = candidates.copy()
        
        # Always take the top result first
        if remaining:
            selected.append(remaining.pop(0))
        
        # For remaining slots, balance relevance with diversity
        while len(selected) < k and remaining:
            best_idx = 0
            best_score = -1
            
            for i, candidate in enumerate(remaining):
                # Relevance score (cosine similarity)
                relevance_score = candidate['score']
                
                # Diversity score (check against already selected)
                diversity_score = 1.0  # Start with max diversity
                for selected_doc in selected:
                    # Simple content-based diversity check
                    if self._content_similarity(candidate['content'], selected_doc['content']) > 0.7:
                        diversity_score *= 0.5  # Reduce diversity score for similar content
                
                # MMR score
                mmr_score = lambda_mult * relevance_score + (1 - lambda_mult) * diversity_score
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        search_time = (time.time() - start_time) * 1000
        return selected, search_time
    
    def _content_similarity(self, text1: str, text2: str) -> float:
        """Simple content similarity based on word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        return overlap / total if total > 0 else 0.0
    
    def filtered_search(self, query: str, k: int = 5) -> Tuple[List[Dict], float]:
        """Search with section-based filtering (mimics your rag.py filter approach)."""
        start_time = time.time()
        
        # Extract keyword for filtering (simplified version of your approach)
        keyword = query.lower().split()[0] if query.strip() else ""
        
        # Embed the query
        query_vector = self.embed_query(query)
        
        # Create filter (look for keyword in title or content)
        search_filter = None
        if len(keyword) > 2:  # Only filter for meaningful keywords
            search_filter = Filter(
                should=[  # Use 'should' instead of 'must' for more flexible filtering
                    FieldCondition(
                        key="title",
                        match=MatchText(text=keyword)
                    ),
                    FieldCondition(
                        key="section",
                        match=MatchText(text=keyword)
                    )
                ]
            )
        
        # Search with filter
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=k,
                with_payload=True
            )
        except:
            # Fallback to no filter if filtering fails
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
                with_payload=True
            )
        
        search_time = (time.time() - start_time) * 1000
        
        # Format results
        results = []
        for point in search_result:
            results.append({
                'id': str(point.id),
                'content': point.payload.get('text', ''),
                'title': point.payload.get('title', ''),
                'file_path': point.payload.get('file_path', ''),
                'section': point.payload.get('section', ''),
                'subsection': point.payload.get('subsection', ''),
                'score': float(point.score),
                'method': f'Filtered ({keyword})'
            })
        
        return results, search_time
    
    def test_query(self, query: str) -> Dict:
        """Test all retrieval methods on a single query."""
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        methods = {
            'Pure Similarity': self.similarity_search,
            'MMR (Relevance + Diversity)': self.mmr_search,
            'Filtered Search': self.filtered_search
        }
        
        results = {}
        
        for method_name, method_func in methods.items():
            try:
                docs, search_time = method_func(query)
                results[method_name] = {
                    'documents': docs,
                    'search_time_ms': search_time,
                    'num_results': len(docs)
                }
                
                # Show results
                print(f"\n{method_name}:")
                print(f"  Time: {search_time:.1f}ms, Results: {len(docs)}")
                
                for i, doc in enumerate(docs[:3], 1):  # Show top 3
                    title = doc['title'][:50] + "..." if len(doc['title']) > 50 else doc['title']
                    section = f" [{doc['section']}]" if doc['section'] else ""
                    score = f" (score: {doc['score']:.3f})" if 'score' in doc else ""
                    print(f"    {i}. {title}{section}{score}")
                
            except Exception as e:
                print(f"\n{method_name}: ERROR - {str(e)}")
                results[method_name] = None
        
        return results
    
    def analyze_result_overlap(self, results: Dict) -> None:
        """Analyze overlap between different methods."""
        print(f"\nOverlap Analysis:")
        print("-" * 30)
        
        valid_methods = {k: v for k, v in results.items() if v and v['documents']}
        method_names = list(valid_methods.keys())
        
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                docs1_ids = {doc['id'] for doc in valid_methods[method1]['documents']}
                docs2_ids = {doc['id'] for doc in valid_methods[method2]['documents']}
                
                overlap = len(docs1_ids & docs2_ids)
                total_unique = len(docs1_ids | docs2_ids)
                
                print(f"{method1} vs {method2}: {overlap} shared docs")
    
    def run_comprehensive_evaluation(self) -> None:
        """Run evaluation across all test queries."""
        print("="*70)
        print("DIRECT QDRANT RETRIEVAL EVALUATION")
        print("="*70)
        print(f"Collection: {self.collection_name}")
        print(f"Embedding: all-MiniLM-L6-v2")
        print(f"Test queries: {len(self.test_queries)}")
        
        # Test each query
        all_results = {}
        method_performance = {
            'Pure Similarity': {'times': [], 'result_counts': []},
            'MMR (Relevance + Diversity)': {'times': [], 'result_counts': []},
            'Filtered Search': {'times': [], 'result_counts': []}
        }
        
        for i, query in enumerate(self.test_queries[:5], 1):  # Test first 5 queries
            print(f"\n[{i}/5] " + "="*50)
            query_results = self.test_query(query)
            all_results[query] = query_results
            
            # Collect performance stats
            for method_name, result in query_results.items():
                if result and method_name in method_performance:
                    method_performance[method_name]['times'].append(result['search_time_ms'])
                    method_performance[method_name]['result_counts'].append(result['num_results'])
            
            # Show overlap
            self.analyze_result_overlap(query_results)
        
        # Summary statistics
        print(f"\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        
        for method_name, stats in method_performance.items():
            if stats['times']:
                avg_time = np.mean(stats['times'])
                avg_results = np.mean(stats['result_counts'])
                print(f"\n{method_name}:")
                print(f"  Average time: {avg_time:.1f}ms")
                print(f"  Average results: {avg_results:.1f} docs")
                print(f"  Queries completed: {len(stats['times'])}")
        
        print(f"\n" + "="*70)


def main():
    """Run the direct retrieval evaluation."""
    tester = DirectRetrievalTester()
    tester.run_comprehensive_evaluation()


if __name__ == "__main__":
    main()
