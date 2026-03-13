"""
RAG Agent (Retrieval-Augmented Generation)
Searches medical knowledge base and provides context for responses
This implements "datos reales" requirement from TFM
"""

from typing import List, Dict, Any, Optional
import time
from logger import HealthChatLogger


class RAGAgent:
    """
    RAG Agent - Retrieves relevant medical information from knowledge base
    Provides context-grounded responses based on real data
    """
    
    def __init__(self, knowledge_base: List[Dict]):
        self.knowledge_base = knowledge_base
        self.logger = HealthChatLogger()
    
    def search(self, query: str, language: str = "es", top_k: int = 3) -> Dict[str, Any]:
        """
        Search knowledge base for relevant information
        
        Args:
            query: User's question
            language: Language code (es, en, fr)
            top_k: Number of results to return
        
        Returns:
            Dict with results and metadata
        """
        start_time = time.time()
        
        # Normalize inputs
        query_lower = query.lower()
        lang = self._normalize_language(language)
        
        # Search knowledge base
        results = self._semantic_search(query_lower, lang, top_k)
        
        # Calculate search time
        search_time = time.time() - start_time
        
        # Log search
        self.logger.log_metrics(
            "rag_search_time",
            search_time,
            {
                "language": lang,
                "results_found": len(results),
                "query_length": len(query)
            }
        )
        
        return {
            "results": results,
            "query": query,
            "language": lang,
            "results_count": len(results),
            "search_time": search_time,
            "has_context": len(results) > 0
        }
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language code"""
        lang_map = {
            "español": "es",
            "spanish": "es",
            "es": "es",
            "english": "en",
            "en": "en",
            "français": "fr",
            "french": "fr",
            "fr": "fr"
        }
        return lang_map.get(language.lower(), "es")
    
    def _semantic_search(self, query: str, language: str, top_k: int) -> List[Dict]:
        """
        Semantic search using keyword matching and relevance scoring
        In production, this would use vector embeddings
        """
        results = []
        
        for entry in self.knowledge_base:
            # Skip if language not available
            if language not in entry:
                continue
            
            question = entry[language].get("question", "").lower()
            answer = entry[language].get("answer", "").lower()
            topic = entry.get("topic", "").lower()
            
            # Calculate relevance score
            score = self._calculate_relevance(query, question, answer, topic)
            
            if score > 0:
                results.append({
                    "id": entry["id"],
                    "category": entry["category"],
                    "topic": entry["topic"],
                    "question": entry[language]["question"],
                    "answer": entry[language]["answer"],
                    "relevance_score": score
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results[:top_k]
    
    def _calculate_relevance(self, query: str, question: str, answer: str, topic: str) -> float:
        """
        Calculate relevance score using keyword matching
        More sophisticated than simple word count
        """
        score = 0.0
        
        # Extract query keywords (remove common words)
        stopwords = {'de', 'la', 'el', 'en', 'y', 'a', 'para', 'por', 'con', 'un', 'una',
                     'the', 'is', 'in', 'to', 'and', 'a', 'of', 'for', 'on', 'with',
                     'le', 'la', 'de', 'et', 'un', 'une', 'pour', 'dans'}
        
        query_words = [w for w in query.split() if w not in stopwords and len(w) > 2]
        
        # Score based on keyword matches
        for word in query_words:
            # Question match (highest weight)
            if word in question:
                score += 3.0
            
            # Topic match (high weight)
            if word in topic:
                score += 2.5
            
            # Answer match (medium weight)
            if word in answer:
                score += 1.0
        
        # Bonus for multiple word matches
        matches = sum(1 for word in query_words if word in question or word in answer)
        if matches > 1:
            score += matches * 0.5
        
        return score
    
    def get_best_match(self, query: str, language: str = "es") -> Optional[Dict]:
        """
        Get single best matching result
        Returns None if no good match found
        """
        results = self.search(query, language, top_k=1)
        
        if results["results"] and results["results"][0]["relevance_score"] > 2.0:
            return results["results"][0]
        
        return None
    
    def format_context(self, search_results: List[Dict], max_length: int = 500) -> str:
        """
        Format search results into context string for LLM
        """
        if not search_results:
            return ""
        
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context = f"[Información {i}]\n"
            context += f"Tema: {result['topic']}\n"
            context += f"Contenido: {result['answer']}\n"
            context_parts.append(context)
        
        full_context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(full_context) > max_length:
            full_context = full_context[:max_length] + "..."
        
        return full_context
    
    def has_relevant_info(self, query: str, language: str = "es", threshold: float = 2.0) -> bool:
        """
        Check if knowledge base has relevant information for query
        """
        best_match = self.get_best_match(query, language)
        return best_match is not None and best_match["relevance_score"] >= threshold
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about knowledge base coverage"""
        total = len(self.knowledge_base)
        
        categories = {}
        languages = set()
        
        for entry in self.knowledge_base:
            cat = entry.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            
            for key in entry.keys():
                if key not in ["id", "category", "topic"]:
                    languages.add(key)
        
        return {
            "total_entries": total,
            "categories": categories,
            "languages": list(languages),
            "topics_covered": [entry.get("topic") for entry in self.knowledge_base]
        }


# Example usage
if __name__ == "__main__":
    # Import knowledge base
    from agents.knowledge_base import MEDICAL_KNOWLEDGE_BASE
    
    # Initialize RAG Agent
    rag_agent = RAGAgent(MEDICAL_KNOWLEDGE_BASE)
    
    print("=" * 70)
    print("RAG AGENT - KNOWLEDGE RETRIEVAL TESTING")
    print("=" * 70)
    
    # Statistics
    stats = rag_agent.get_statistics()
    print(f"\nKnowledge Base Statistics:")
    print(f"Total Entries: {stats['total_entries']}")
    print(f"Languages: {', '.join(stats['languages'])}")
    print(f"Categories: {stats['categories']}")
    
    print("\n" + "=" * 70)
    print("TESTING SEARCH QUERIES")
    print("=" * 70)
    
    # Test queries
    test_queries = [
        ("¿Por qué me duele la cabeza?", "es"),
        ("How to reduce stress?", "en"),
        ("todo me duele", "es"),
        ("Comment bien dormir?", "fr"),
        ("What foods are healthy?", "en"),
        ("¿Cómo controlar la diabetes?", "es"),
    ]
    
    for query, lang in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}' (Language: {lang})")
        print(f"{'='*70}")
        
        # Search
        search_results = rag_agent.search(query, lang, top_k=2)
        
        print(f"\nResults Found: {search_results['results_count']}")
        print(f"Search Time: {search_results['search_time']:.3f}s")
        print(f"Has Context: {search_results['has_context']}")
        
        if search_results['results']:
            print(f"\nTop Results:")
            for i, result in enumerate(search_results['results'], 1):
                print(f"\n  [{i}] Topic: {result['topic']}")
                print(f"      Relevance: {result['relevance_score']:.1f}")
                print(f"      Q: {result['question']}")
                print(f"      A: {result['answer'][:150]}...")
        else:
            print("\n  No relevant results found in knowledge base")
        
        # Check if has relevant info
        has_info = rag_agent.has_relevant_info(query, lang)
        print(f"\n  ✅ Has relevant KB info: {has_info}")
    
    print("\n" + "=" * 70)
    print("RAG AGENT TEST COMPLETE")
    print("=" * 70)