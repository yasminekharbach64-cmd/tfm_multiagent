"""
RAG Agent (Retrieval-Augmented Generation) — PubMed Edition
Searches PubMed scientific papers and provides context + citations
"""

from typing import List, Dict, Any, Optional
import time
from logger import HealthChatLogger


class RAGAgent:
    """
    RAG Agent - Retrieves relevant PubMed papers for user queries
    Returns ranked scientific citations with abstracts
    """

    def __init__(self, knowledge_base: List[Dict]):
        self.knowledge_base = knowledge_base
        self.logger = HealthChatLogger()

        
        self.KEYWORD_TRANSLATIONS = {
           
            "fatiga": "fatigue", "cansancio": "fatigue", "cansada": "fatigue",
            "dolor": "pain", "dolores": "pain",
            "sueño": "sleep", "insomnio": "insomnia", "dormir": "sleep",
            "ansiedad": "anxiety", "estrés": "stress", "estres": "stress",
            "depresión": "depression", "depresion": "depression", "triste": "depression",
            "memoria": "memory", "cognitivo": "cognitive", "cognición": "cognition",
            "concentración": "concentration", "olvidos": "memory",
            "humor": "mood", "ánimo": "mood", "animo": "mood",
            "irritabilidad": "irritability", "irritable": "irritable",
            "sofocos": "hot flashes", "sofoco": "hot flashes", "calores": "hot flashes",
            "sudoración": "sweating", "sudores": "sweating",
            "libido": "libido", "sexual": "sexual", "sexualidad": "sexuality",
            "sequedad": "dryness", "vaginal": "vaginal",
            "peso": "weight", "obesidad": "obesity", "adelgazar": "weight loss",
            "huesos": "bone", "osteoporosis": "osteoporosis", "fractura": "fracture",
            "corazón": "cardiovascular", "presión": "blood pressure",
            "tiroides": "thyroid", "hormonas": "hormones", "hormonal": "hormonal",
            "menopausia": "menopause", "menopáusica": "menopausal",
            "perimenopausia": "perimenopause", "climaterio": "menopause",
            "tratamiento": "treatment", "terapia": "therapy",
            "ejercicio": "exercise", "actividad": "activity",
            "dieta": "diet", "nutrición": "nutrition",
            "calidad": "quality", "vida": "life",
            # Français
            "fatigue": "fatigue", "douleur": "pain", "sommeil": "sleep",
            "anxiété": "anxiety", "stress": "stress", "dépression": "depression",
            "mémoire": "memory", "humeur": "mood", "bouffées": "hot flashes",
            "ménopause": "menopause", "traitement": "treatment",
        }

    

    def search(self, query: str, language: str = "es", top_k: int = 3) -> Dict[str, Any]:
        """
        Search PubMed knowledge base for relevant papers

        Args:
            query: User's question
            language: Language code (es, en, fr) — not used for filtering,
                      papers are in English but context is language-agnostic
            top_k: Number of top papers to return

        Returns:
            Dict with results, citations, and metadata
        """
        start_time = time.time()

        query_lower = query.lower()
        lang = self._normalize_language(language)

        
        query_translated = self._translate_query(query_lower)

        
        results = self._semantic_search(query_lower, top_k, query_translated)

        search_time = time.time() - start_time

        
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
            "has_context": len(results) > 0,
            "citations": self._format_citations(results, threshold=7.0)
        }

    

    def _translate_query(self, query: str) -> str:
        """Traduce keywords médicos español/français → english"""
        words = query.split()
        return " ".join(self.KEYWORD_TRANSLATIONS.get(w, w) for w in words)

    def _semantic_search(self, query: str, top_k: int, query_translated: str = "") -> List[Dict]:
        """
        Search PubMed papers — usa query original + traducida para cobertura multilingüe
        """
        results = []

        for paper in self.knowledge_base:
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()

            score = self._calculate_relevance(query, title, abstract)

            
            if query_translated and query_translated != query:
                score_translated = self._calculate_relevance(query_translated, title, abstract)
                score = max(score, score_translated)

            if score > 0:
                results.append({
                    
                    "pmid": paper.get("pmid", ""),
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "authors_str": paper.get("authors_str", ""),
                    "year": paper.get("year", ""),
                    "journal": paper.get("journal", ""),
                    "url": paper.get("url", ""),
                    "citation": paper.get("citation", ""),
                    
                    "relevance_score": score,
                    
                    "answer": paper.get("abstract", "")[:300],
                    "topic": paper.get("title", "")[:60],
                    "category": "pubmed_paper",
                })

        
        for r in results:
            try:
                year_bonus = (int(r["year"]) - 2020) * 0.1
                r["relevance_score"] += year_bonus
            except (ValueError, TypeError):
                pass

        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        if results and results[0]["relevance_score"] < 7.0:
            return []

        return results[:top_k]

    

    def _calculate_relevance(self, query: str, title: str, abstract: str) -> float:
        """
        Relevance score basado en keyword matching
        Title match pesa más que abstract match
        """
        score = 0.0

        stopwords = {
            'de', 'la', 'el', 'en', 'y', 'a', 'para', 'por', 'con', 'un', 'una',
            'que', 'es', 'se', 'no', 'si', 'lo', 'le', 'me', 'mi', 'su', 'al',
            'the', 'is', 'in', 'to', 'and', 'of', 'for', 'on', 'with', 'are',
            'was', 'were', 'has', 'have', 'had', 'this', 'that', 'from', 'by',
            'le', 'et', 'un', 'une', 'pour', 'dans', 'du', 'des', 'les', 'je',
            'i', 'my', 'me', 'we', 'our', 'you', 'it', 'its', 'be', 'at', 'an'
        }

        query_words = [w for w in query.split() if w not in stopwords and len(w) > 2]

        if not query_words:
            return 0.0

        for word in query_words:
            
            if word in title:
                score += 3.0

            
            if word in abstract:
                score += 1.0

            
            if any(word in t_word for t_word in title.split()):
                score += 0.5

        
        matches = sum(1 for w in query_words if w in title or w in abstract)
        if matches > 1:
            score += matches * 0.3

        return score

    

    def _format_citations(self, results: List[Dict], threshold: float = 7.0) -> str:
        """
        Formatea los papers como citations numeradas para mostrar al usuario
        Solo muestra papers con score > threshold (relevancia suficiente)
        """
        if not results:
            return ""

        
        relevant = [r for r in results if r.get("relevance_score", 0) >= threshold]

        if not relevant:
            return ""

        lines = ["\n📚 **Referencias científicas:**"]
        for i, paper in enumerate(relevant, 1):
            lines.append(
                f"[{i}] {paper['authors_str']} ({paper['year']}). "
                f"{paper['title'][:80]}{'...' if len(paper['title']) > 80 else ''}. "
                f"{paper['journal']}."
            )
            if paper.get("url"):
                lines.append(f"    🔗 {paper['url']}")

        return "\n".join(lines)

    

    def format_context(self, search_results: List[Dict], max_length: int = 800) -> str:
        """
        Formatea los abstracts como contexto para el LLM
        """
        if not search_results:
            return ""

        parts = []
        for i, paper in enumerate(search_results, 1):
            context = (
                f"[Estudio {i}] {paper['authors_str']} ({paper['year']}) — {paper['journal']}\n"
                f"Título: {paper['title']}\n"
                f"Evidencia: {paper['abstract'][:150]}\n"
            )
            parts.append(context)

        full = "\n".join(parts)
        if len(full) > max_length:
            full = full[:max_length] + "..."
        return full

    

    def _normalize_language(self, language: str) -> str:
        lang_map = {
            "español": "es", "spanish": "es", "es": "es",
            "english": "en", "en": "en",
            "français": "fr", "french": "fr", "fr": "fr"
        }
        return lang_map.get(language.lower(), "es")

    def get_best_match(self, query: str, language: str = "es") -> Optional[Dict]:
        results = self.search(query, language, top_k=1)
        if results["results"] and results["results"][0]["relevance_score"] > 2.0:
            return results["results"][0]
        return None

    def has_relevant_info(self, query: str, language: str = "es", threshold: float = 2.0) -> bool:
        best_match = self.get_best_match(query, language)
        return best_match is not None and best_match["relevance_score"] >= threshold

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self.knowledge_base),
            "source": "PubMed (NCBI)",
            "type": "scientific_papers"
        }




if __name__ == "__main__":
    from agents.knowledge_base import MEDICAL_KNOWLEDGE_BASE

    rag = RAGAgent(MEDICAL_KNOWLEDGE_BASE)

    print("=" * 70)
    print("RAG AGENT — PubMed Edition TEST")
    print("=" * 70)

    test_queries = [
        ("fatiga y menopausia", "es"),
        ("sleep disorders menopause", "en"),
        ("ansiedad menopausia tratamiento", "es"),
        ("cognitive dysfunction menopause", "en"),
    ]

    for query, lang in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print(f"{'='*70}")

        res = rag.search(query, lang, top_k=3)

        print(f"Results: {res['results_count']} | Time: {res['search_time']:.3f}s")

        for i, paper in enumerate(res["results"], 1):
            print(f"\n  [{i}] Score: {paper['relevance_score']:.1f}")
            print(f"      {paper['authors_str']} ({paper['year']})")
            print(f"      {paper['title'][:70]}...")
            print(f"      Journal: {paper['journal']}")
            print(f"      Abstract: {paper['abstract'][:120]}...")

        print(res["citations"])

    print("\n" + "=" * 70)
    print("✅ RAG AGENT TEST COMPLETE")
    print("=" * 70)