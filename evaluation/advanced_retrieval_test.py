"""
Advanced retrieval method testing to find the optimal approach.
Tests more sophisticated methods including hybrid search and different parameters.
"""

import time
import numpy as np
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from collections import defaultdict, Counter
import re


class AdvancedRetrievalTester:
    """Test advanced retrieval methods to find the optimal approach."""
    
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "godot-docs"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # More diverse test queries
        self.test_queries = [
            "how to move a 2D character with input",  # Movement
            "add camera follow player",              # Camera
            "create button UI interface",           # UI
            "collision detection physics",          # Physics
            "export game build release",           # Export
            "play sound audio effects",           # Audio
            "load save game data",                # Data persistence
            "animation sprite frames",            # Animation
            "scene management switching",         # Scene management
            "shader material effects"            # Advanced graphics
        ]
    
    def embed_query(self, query: str) -> List[float]:
        """Embed query using same model as indexing."""
        return self.embedding_model.embed_query(query)
    
    def pure_similarity_search(self, query: str, k: int = 5) -> Tuple[List[Dict], float]:
        """Pure cosine similarity search - your baseline."""
        start_time = time.time()
        
        query_vector = self.embed_query(query)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True
        )
        
        search_time = (time.time() - start_time) * 1000
        
        results = []
        for point in search_result:
            results.append({
                'id': str(point.id),
                'content': point.payload.get('text', ''),
                'title': point.payload.get('title', ''),
                'section': point.payload.get('section', ''),
                'score': float(point.score),
                'method': 'Pure_Similarity'
            })
        
        return results, search_time
    
    def aggressive_mmr_search(self, query: str, k: int = 5) -> Tuple[List[Dict], float]:
        """MMR with more aggressive diversity (lower lambda)."""
        start_time = time.time()
        
        # Get many more candidates for better MMR selection
        candidates_k = k * 4
        query_vector = self.embed_query(query)
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=candidates_k,
            with_payload=True
        )
        
        if not search_result:
            return [], (time.time() - start_time) * 1000
        
        candidates = []
        for point in search_result:
            candidates.append({
                'id': str(point.id),
                'content': point.payload.get('text', ''),
                'title': point.payload.get('title', ''),
                'section': point.payload.get('section', ''),
                'score': float(point.score),
                'method': 'Aggressive_MMR'
            })
        
        # Aggressive MMR selection (lambda = 0.3, favoring diversity)
        selected = []
        remaining = candidates.copy()
        
        if remaining:
            selected.append(remaining.pop(0))  # Always take top result
        
        while len(selected) < k and remaining:
            best_idx = 0
            best_score = -float('inf')
            
            for i, candidate in enumerate(remaining):
                relevance = candidate['score']
                
                # Strong diversity penalty
                diversity = 1.0
                for selected_doc in selected:
                    # Section diversity
                    if candidate['section'] == selected_doc['section']:
                        diversity *= 0.3
                    
                    # Title similarity
                    title_sim = self._text_similarity(candidate['title'], selected_doc['title'])
                    if title_sim > 0.5:
                        diversity *= 0.4
                    
                    # Content diversity
                    content_sim = self._text_similarity(candidate['content'][:200], selected_doc['content'][:200])
                    if content_sim > 0.6:
                        diversity *= 0.5
                
                # MMR score with aggressive diversity (lambda=0.3)
                mmr_score = 0.3 * relevance + 0.7 * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        search_time = (time.time() - start_time) * 1000
        return selected, search_time
    
    def section_diversified_search(self, query: str, k: int = 5) -> Tuple[List[Dict], float]:
        """Ensure results come from different documentation sections."""
        start_time = time.time()
        
        query_vector = self.embed_query(query)
        
        # Get more candidates
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k * 3,
            with_payload=True
        )
        
        if not search_result:
            return [], (time.time() - start_time) * 1000
        
        # Group by section
        by_section = defaultdict(list)
        for point in search_result:
            section = point.payload.get('section', 'unknown')
            by_section[section].append({
                'id': str(point.id),
                'content': point.payload.get('text', ''),
                'title': point.payload.get('title', ''),
                'section': section,
                'score': float(point.score),
                'method': 'Section_Diversified'
            })
        
        # Select best from each section
        selected = []
        sections_used = set()
        
        # First pass: one from each section
        for section, docs in by_section.items():
            if len(selected) < k:
                selected.append(docs[0])  # Best from this section
                sections_used.add(section)
        
        # Second pass: fill remaining slots with best overall
        all_remaining = []
        for section, docs in by_section.items():
            all_remaining.extend(docs[1:])  # Skip the first one we already took
        
        all_remaining.sort(key=lambda x: x['score'], reverse=True)
        
        for doc in all_remaining:
            if len(selected) >= k:
                break
            selected.append(doc)
        
        search_time = (time.time() - start_time) * 1000
        return selected[:k], search_time
    
    def keyword_boosted_search(self, query: str, k: int = 5) -> Tuple[List[Dict], float]:
        """Boost results that contain query keywords in title/section."""
        start_time = time.time()
        
        query_vector = self.embed_query(query)
        
        # Extract keywords from query
        keywords = [word.lower() for word in re.findall(r'\b\w+\b', query) if len(word) > 2]
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k * 2,
            with_payload=True
        )
        
        results = []
        for point in search_result:
            doc = {
                'id': str(point.id),
                'content': point.payload.get('text', ''),
                'title': point.payload.get('title', ''),
                'section': point.payload.get('section', ''),
                'score': float(point.score),
                'method': 'Keyword_Boosted'
            }
            
            # Boost score if keywords appear in title or section
            boost_factor = 1.0
            title_lower = doc['title'].lower()
            section_lower = doc['section'].lower()
            
            for keyword in keywords:
                if keyword in title_lower:
                    boost_factor += 0.1
                if keyword in section_lower:
                    boost_factor += 0.05
            
            doc['score'] *= boost_factor
            results.append(doc)
        
        # Re-sort by boosted scores
        results.sort(key=lambda x: x['score'], reverse=True)
        
        search_time = (time.time() - start_time) * 1000
        return results[:k], search_time
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity based on word overlap."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        return overlap / total
    
    def evaluate_method_quality(self, results: List[Dict], query: str) -> Dict:
        """Evaluate the quality of results for a given query."""
        if not results:
            return {'relevance_score': 0, 'diversity_score': 0, 'section_coverage': 0}
        
        # Relevance: average of top 3 scores
        top_scores = [doc['score'] for doc in results[:3]]
        relevance_score = np.mean(top_scores) if top_scores else 0
        
        # Diversity: how different are the sections?
        sections = [doc['section'] for doc in results]
        unique_sections = len(set(sections))
        section_coverage = unique_sections / len(results) if results else 0
        
        # Title diversity
        titles = [doc['title'] for doc in results]
        title_similarities = []
        for i in range(len(titles)):
            for j in range(i+1, len(titles)):
                sim = self._text_similarity(titles[i], titles[j])
                title_similarities.append(sim)
        
        diversity_score = 1 - (np.mean(title_similarities) if title_similarities else 0)
        
        return {
            'relevance_score': relevance_score,
            'diversity_score': diversity_score,
            'section_coverage': section_coverage
        }
    
    def run_comprehensive_comparison(self) -> None:
        """Run comprehensive comparison of all methods."""
        print("="*80)
        print("ADVANCED RETRIEVAL METHOD COMPARISON")
        print("="*80)
        
        methods = {
            'Pure Similarity': self.pure_similarity_search,
            'Aggressive MMR': self.aggressive_mmr_search,
            'Section Diversified': self.section_diversified_search,
            'Keyword Boosted': self.keyword_boosted_search
        }
        
        method_stats = defaultdict(lambda: {
            'times': [],
            'relevance_scores': [],
            'diversity_scores': [],
            'section_coverage': []
        })
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"\n[{i}/{len(self.test_queries)}] Query: '{query}'")
            print("-" * 60)
            
            query_results = {}
            
            for method_name, method_func in methods.items():
                try:
                    docs, search_time = method_func(query)
                    quality = self.evaluate_method_quality(docs, query)
                    
                    # Store stats
                    method_stats[method_name]['times'].append(search_time)
                    method_stats[method_name]['relevance_scores'].append(quality['relevance_score'])
                    method_stats[method_name]['diversity_scores'].append(quality['diversity_score'])
                    method_stats[method_name]['section_coverage'].append(quality['section_coverage'])
                    
                    # Show top result
                    if docs:
                        top_doc = docs[0]
                        sections = list(set([doc['section'] for doc in docs]))
                        print(f"{method_name:20} | {search_time:5.1f}ms | Score: {quality['relevance_score']:.3f} | Sections: {len(sections)} | Top: {top_doc['title'][:40]}...")
                    
                except Exception as e:
                    print(f"{method_name:20} | ERROR: {str(e)}")
        
        # Final comparison
        print(f"\n" + "="*80)
        print("OVERALL PERFORMANCE COMPARISON")
        print("="*80)
        print(f"{'Method':<20} | {'Avg Time':<10} | {'Relevance':<10} | {'Diversity':<10} | {'Coverage':<10}")
        print("-" * 80)
        
        method_scores = {}
        for method_name, stats in method_stats.items():
            if stats['times']:
                avg_time = np.mean(stats['times'])
                avg_relevance = np.mean(stats['relevance_scores'])
                avg_diversity = np.mean(stats['diversity_scores'])
                avg_coverage = np.mean(stats['section_coverage'])
                
                # Combined score (relevance=50%, diversity=30%, coverage=20%, speed bonus)
                speed_bonus = max(0, (50 - avg_time) / 1000)  # Bonus for being under 50ms
                combined_score = (avg_relevance * 0.5 + avg_diversity * 0.3 + avg_coverage * 0.2 + speed_bonus)
                method_scores[method_name] = combined_score
                
                print(f"{method_name:<20} | {avg_time:8.1f}ms | {avg_relevance:8.3f} | {avg_diversity:8.3f} | {avg_coverage:8.3f}")
        
        # Winner
        if method_scores:
            winner = max(method_scores.keys(), key=lambda x: method_scores[x])
            print(f"\n🏆 WINNER: {winner} (Combined Score: {method_scores[winner]:.3f})")
            


def main():
    """Run the advanced retrieval comparison."""
    tester = AdvancedRetrievalTester()
    tester.run_comprehensive_comparison()


if __name__ == "__main__":
    main()
