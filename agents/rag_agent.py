
from typing import List, Dict, Any, Optional, Tuple
import os
import sys
import json
import time
import re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import HealthChatLogger



EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), "pubmed_embeddings.npy")
META_FILE = os.path.join(os.path.dirname(__file__), "pubmed_embeddings_meta.json")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class RAGAgent:
    """
    Hybrid RAG agent for female hormonal health PubMed papers.
    Combines semantic search (embeddings) with keyword matching.
    """

   
    MIN_SCORE_TO_INCLUDE = 0.30    
    MIN_SCORE_FOR_CITATION = 0.45 
    HIGH_CONFIDENCE_THRESHOLD = 0.55 

   
    WEIGHT_SEMANTIC = 0.7
    WEIGHT_KEYWORD = 0.3

    def __init__(self, knowledge_base: List[Dict]):
        self.knowledge_base = knowledge_base
        self.logger = HealthChatLogger()

        
        self.embeddings: Optional[np.ndarray] = None
        self.model = None
        self.semantic_enabled = False

        self._setup_keyword_mappings()
        self._load_embeddings()

    
    def _load_embeddings(self) -> None:
        """Load precomputed embeddings + sentence-transformers model."""
        if not os.path.exists(EMBEDDINGS_FILE):
            print(f"⚠️  RAG: embeddings file not found at {EMBEDDINGS_FILE}")
            print("⚠️  RAG: falling back to KEYWORD-ONLY mode")
            print("⚠️  RAG: run `python build_embeddings.py` to enable semantic search")
            return

        if not os.path.exists(META_FILE):
            print(f"⚠️  RAG: metadata file not found at {META_FILE}")
            print("⚠️  RAG: falling back to KEYWORD-ONLY mode")
            return

        try:
            
            with open(META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("num_papers") != len(self.knowledge_base):
                print(f"⚠️  RAG: embeddings count ({meta.get('num_papers')}) "
                      f"≠ KB count ({len(self.knowledge_base)})")
                print("⚠️  RAG: rebuild needed — run `python build_embeddings.py`")
                print("⚠️  RAG: falling back to KEYWORD-ONLY mode")
                return

           
            self.embeddings = np.load(EMBEDDINGS_FILE)

           
            print(f"🤖 RAG: loading embedding model {MODEL_NAME}...")
            t0 = time.time()
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MODEL_NAME)
            self.semantic_enabled = True
            print(f"✅ RAG: semantic search enabled "
                  f"({self.embeddings.shape[0]} papers, "
                  f"{self.embeddings.shape[1]}-dim, "
                  f"loaded in {time.time() - t0:.1f}s)")

        except ImportError:
            print("⚠️  RAG: sentence-transformers not installed")
            print("⚠️  RAG: run `pip install sentence-transformers`")
            print("⚠️  RAG: falling back to KEYWORD-ONLY mode")
        except Exception as e:
            print(f"⚠️  RAG: error loading embeddings — {e}")
            print("⚠️  RAG: falling back to KEYWORD-ONLY mode")

   
    def _setup_keyword_mappings(self) -> None:
        self.PHRASE_TRANSLATIONS = {
            # Spanish → English
            "calidad de vida": "quality of life",
            "salud mental": "mental health",
            "salud ósea": "bone health",
            "ciclo menstrual": "menstrual cycle",
            "síndrome premenstrual": "premenstrual syndrome",
            "trastorno disfórico premenstrual": "premenstrual dysphoric disorder",
            "terapia hormonal": "hormone therapy",
            "terapia de reemplazo hormonal": "hormone replacement therapy",
            "sequedad vaginal": "vaginal dryness",
            "sudoración nocturna": "night sweats",
            "sofocos": "hot flashes",
            "bochornos": "hot flashes",
            "salud sexual": "sexual health",
            "función cognitiva": "cognitive function",
            "deterioro cognitivo": "cognitive decline",
            "perimenopausia": "perimenopause",
            "postmenopausia": "postmenopause",
            "posmenopausia": "postmenopause",
            "endometriosis": "endometriosis",
            "ovario poliquístico": "polycystic ovary",
            "miomas uterinos": "uterine fibroids",
            "fatiga crónica": "chronic fatigue",
            "trastorno del sueño": "sleep disorder",
            "insomnio menopáusico": "menopausal insomnia",
            "salud cardiovascular": "cardiovascular health",
            "densidad ósea": "bone density",
            "salud tiroidea": "thyroid health",
            "hipotiroidismo": "hypothyroidism",
            "hipertiroidismo": "hyperthyroidism",
            # French → English
            "qualité de vie": "quality of life",
            "santé mentale": "mental health",
            "cycle menstruel": "menstrual cycle",
            "syndrome prémenstruel": "premenstrual syndrome",
            "thérapie hormonale": "hormone therapy",
            "bouffées de chaleur": "hot flashes",
            "sécheresse vaginale": "vaginal dryness",
            "sueurs nocturnes": "night sweats",
            "fonction cognitive": "cognitive function",
        }

        self.KEYWORD_TRANSLATIONS = {
            # Spanish
            "fatiga": "fatigue", "cansancio": "fatigue", "cansada": "fatigue",
            "agotamiento": "fatigue", "agotada": "fatigue",
            "dolor": "pain", "dolores": "pain",
            "sueño": "sleep", "insomnio": "insomnia", "dormir": "sleep",
            "ansiedad": "anxiety", "estrés": "stress", "estres": "stress",
            "depresión": "depression", "depresion": "depression",
            "triste": "sad", "tristeza": "sadness",
            "memoria": "memory", "cognitivo": "cognitive", "cognición": "cognition",
            "concentración": "concentration", "olvidos": "memory",
            "humor": "mood", "ánimo": "mood", "animo": "mood",
            "irritabilidad": "irritability", "irritable": "irritable",
            "sofocos": "hot flashes", "sofoco": "hot flashes", "calores": "hot flashes",
            "bochornos": "hot flashes",
            "sudoración": "sweating", "sudores": "sweating",
            "libido": "libido", "sexual": "sexual", "sexualidad": "sexuality",
            "sequedad": "dryness", "vaginal": "vaginal",
            "peso": "weight", "obesidad": "obesity",
            "huesos": "bone", "osteoporosis": "osteoporosis", "fractura": "fracture",
            "tiroides": "thyroid", "hormonas": "hormones", "hormonal": "hormonal",
            "estrógeno": "estrogen", "estrogeno": "estrogen",
            "progesterona": "progesterone",
            "menopausia": "menopause", "menopáusica": "menopausal",
            "perimenopausia": "perimenopause", "climaterio": "menopause",
            "postmenopausia": "postmenopause", "posmenopausia": "postmenopause",
            "tratamiento": "treatment", "terapia": "therapy",
            "ejercicio": "exercise", "actividad": "activity",
            "dieta": "diet", "nutrición": "nutrition",
            "menstruación": "menstruation", "menstruacion": "menstruation",
            "regla": "menstruation", "período": "period", "periodo": "period",
            "ciclo": "cycle", "ovario": "ovary", "ovarios": "ovaries",
            "útero": "uterus", "utero": "uterus", "endometrio": "endometrium",
            "fertilidad": "fertility",
            # French
            "fatigue": "fatigue", "douleur": "pain", "sommeil": "sleep",
            "anxiété": "anxiety", "dépression": "depression",
            "mémoire": "memory", "humeur": "mood",
            "ménopause": "menopause", "périménopause": "perimenopause",
            "traitement": "treatment", "règles": "menstruation",
            "ovaire": "ovary", "utérus": "uterus",
        }

        self.STOPWORDS = {
            'de', 'la', 'el', 'en', 'y', 'a', 'para', 'por', 'con', 'un', 'una',
            'que', 'es', 'se', 'no', 'si', 'lo', 'le', 'me', 'mi', 'su', 'al',
            'tengo', 'tienes', 'tiene', 'qué', 'cómo', 'cuándo', 'dónde',
            'the', 'is', 'in', 'to', 'and', 'of', 'for', 'on', 'with', 'are',
            'was', 'were', 'has', 'have', 'had', 'this', 'that', 'from', 'by',
            'what', 'how', 'when', 'where', 'why', 'do', 'does', 'did',
            'le', 'et', 'pour', 'dans', 'du', 'des', 'les', 'je', 'tu',
            'i', 'my', 'me', 'we', 'our', 'you', 'it', 'its', 'be', 'at', 'an',
        }

   
    def search(self, query: str, language: str = "es", top_k: int = 3) -> Dict[str, Any]:
        """
        Search PubMed knowledge base for relevant papers.
        Returns top_k papers ranked by hybrid (semantic + keyword) relevance.
        """
        start_time = time.time()

        query_lower = query.lower()
        query_clean = re.sub(r'[¿?¡!.,;:()\[\]"\']+', ' ', query_lower)
        query_clean = re.sub(r'\s+', ' ', query_clean).strip()

        lang = self._normalize_language(language)
        query_translated = self._translate_query(query_clean)

        
        if query_translated and query_translated != query_clean:
            query_for_embedding = f"{query_clean}. {query_translated}"
        else:
            query_for_embedding = query_clean

        results = self._hybrid_search(
            query_clean=query_clean,
            query_translated=query_translated,
            query_for_embedding=query_for_embedding,
            top_k=top_k,
        )

        search_time = time.time() - start_time

        self.logger.log_metrics(
            "rag_search_time",
            search_time,
            {
                "language": lang,
                "results_found": len(results),
                "query_length": len(query),
                "top_score": results[0]["relevance_score"] if results else 0,
                "mode": "semantic" if self.semantic_enabled else "keyword_only",
            }
        )

        return {
            "results": results,
            "query": query,
            "language": lang,
            "results_count": len(results),
            "search_time": search_time,
            "has_context": len(results) > 0,
            "has_high_confidence": (
                len(results) > 0
                and results[0]["relevance_score"] >= self.HIGH_CONFIDENCE_THRESHOLD
            ),
            "top_score": results[0]["relevance_score"] if results else 0.0,
            "citations": self._format_citations(results),
            "search_mode": "semantic" if self.semantic_enabled else "keyword_only",
        }

    
    def _hybrid_search(
        self,
        query_clean: str,
        query_translated: str,
        query_for_embedding: str,
        top_k: int,
    ) -> List[Dict]:
        """
        Two-stage retrieval:
        1. SEMANTIC: get top-N candidates by cosine similarity (fast vector search)
        2. RERANK: combine with keyword score → final hybrid score
        """
        
        if self.semantic_enabled:
            query_embedding = self.model.encode(
                [query_for_embedding],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]
           
            cosine_scores = self.embeddings @ query_embedding  
        else:
            
            cosine_scores = np.zeros(len(self.knowledge_base))

        
        n_candidates = min(50, len(self.knowledge_base))
        top_indices = np.argsort(cosine_scores)[-n_candidates:][::-1]

        
        if not self.semantic_enabled:
            top_indices = list(range(len(self.knowledge_base)))

        results = []
        for idx in top_indices:
            paper = self.knowledge_base[idx]
            title = (paper.get("title") or "").lower()
            abstract = (paper.get("abstract") or "").lower()

           
            kw_score_raw = max(
                self._calculate_keyword_score(query_clean, title, abstract),
                self._calculate_keyword_score(query_translated, title, abstract)
                if query_translated and query_translated != query_clean else 0.0,
            )
            
            kw_score_norm = min(kw_score_raw / 15.0, 1.0)

            
            sem_score = float(cosine_scores[idx])

            
            if self.semantic_enabled:
                hybrid_score = (
                    self.WEIGHT_SEMANTIC * sem_score
                    + self.WEIGHT_KEYWORD * kw_score_norm
                )
            else:
                
                hybrid_score = kw_score_norm

            
            try:
                year_bonus = max(0, (int(paper.get("year", 0)) - 2020) * 0.005)
                hybrid_score += year_bonus
            except (ValueError, TypeError):
                pass

            if hybrid_score < self.MIN_SCORE_TO_INCLUDE:
                continue

            confidence = self._confidence_label(hybrid_score)

            results.append({
                "pmid": paper.get("pmid", ""),
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors_str": paper.get("authors_str", ""),
                "year": paper.get("year", ""),
                "journal": paper.get("journal", ""),
                "url": paper.get("url", ""),
                "doi": paper.get("doi", ""),
                "citation": paper.get("citation", ""),
                "relevance_score": hybrid_score,
                "semantic_score": sem_score,
                "keyword_score": kw_score_norm,
                "confidence": confidence,
               
                "answer": (paper.get("abstract") or "")[:300],
                "topic": (paper.get("title") or "")[:60],
                "category": "pubmed_paper",
            })

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

   
    def _calculate_keyword_score(self, query: str, title: str, abstract: str) -> float:
        
        if not query:
            return 0.0

        score = 0.0
        query_words = [w for w in query.split() if w not in self.STOPWORDS and len(w) > 2]
        if not query_words:
            return 0.0

        
        for word in query_words:
            if word in title:
                score += 3.0
            if word in abstract:
                score += 1.0

       
        if len(query_words) >= 2:
            for i in range(len(query_words) - 1):
                bigram = f"{query_words[i]} {query_words[i+1]}"
                if bigram in title:
                    score += 5.0
                elif bigram in abstract:
                    score += 2.0

       
        matching_words = sum(1 for w in query_words if w in title or w in abstract)
        if matching_words >= 2:
            score += matching_words * 0.3

        return score

    
    def _translate_query(self, query: str) -> str:
        translated = query
        sorted_phrases = sorted(self.PHRASE_TRANSLATIONS.keys(), key=len, reverse=True)
        for phrase in sorted_phrases:
            if phrase in translated:
                translated = translated.replace(phrase, self.PHRASE_TRANSLATIONS[phrase])
        words = translated.split()
        translated_words = [self.KEYWORD_TRANSLATIONS.get(w, w) for w in words]
        return " ".join(translated_words)

    
    def _confidence_label(self, score: float) -> str:
        if score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return "alta"
        if score >= self.MIN_SCORE_FOR_CITATION:
            return "media"
        return "baja"


    def _format_citations(self, results: List[Dict]) -> str:
        if not results:
            return ""

        relevant = [r for r in results if r.get("relevance_score", 0) >= self.MIN_SCORE_FOR_CITATION]
        if not relevant:
            return ""

        lines = ["📚 **Referencias científicas (PubMed):**"]
        for i, paper in enumerate(relevant, 1):
            title = (paper.get("title") or "").strip()
            if len(title) > 100:
                title = title[:100] + "..."

            authors = paper.get("authors_str") or "Autores no disponibles"
            year = paper.get("year") or "—"
            journal = paper.get("journal") or ""

            citation_line = f"[{i}] {authors} ({year}). *{title}*"
            if journal:
                citation_line += f". {journal}."
            lines.append(citation_line)

            if paper.get("url"):
                lines.append(f"    🔗 {paper['url']}")

        return "\n".join(lines)


    def format_context(self, search_results: List[Dict], max_length: int = 1000) -> str:
        if not search_results:
            return ""

        parts = []
        for i, paper in enumerate(search_results, 1):
            abstract = (paper.get('abstract') or '')[:200]
            context = (
                f"[Estudio {i}] {paper.get('authors_str', '')} ({paper.get('year', '')}) "
                f"— {paper.get('journal', '')}\n"
                f"Título: {paper.get('title', '')}\n"
                f"Evidencia: {abstract}\n"
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
            "français": "fr", "french": "fr", "fr": "fr",
            "arabic": "ar", "ar": "ar",
        }
        return lang_map.get(language.lower(), "es")

    def get_best_match(self, query: str, language: str = "es") -> Optional[Dict]:
        results = self.search(query, language, top_k=1)
        if results["results"] and results["results"][0]["relevance_score"] > self.MIN_SCORE_TO_INCLUDE:
            return results["results"][0]
        return None

    def has_relevant_info(self, query: str, language: str = "es", threshold: float = None) -> bool:
        if threshold is None:
            threshold = self.MIN_SCORE_TO_INCLUDE
        best_match = self.get_best_match(query, language)
        return best_match is not None and best_match["relevance_score"] >= threshold

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self.knowledge_base),
            "source": "PubMed (NCBI)",
            "type": "scientific_papers",
            "domain": "female_hormonal_health",
            "min_score_threshold": self.MIN_SCORE_TO_INCLUDE,
            "search_mode": "semantic" if self.semantic_enabled else "keyword_only",
            "embedding_model": MODEL_NAME if self.semantic_enabled else None,
        }



if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.knowledge_base import MEDICAL_KNOWLEDGE_BASE

    rag = RAGAgent(MEDICAL_KNOWLEDGE_BASE)

    print("\n" + "=" * 70)
    print("RAG AGENT v3.0 — HYBRID SEMANTIC + KEYWORD TEST")
    print("=" * 70)

  
    test_queries = [
        ("¿Por qué tengo tanto calor de repente?", "es"),
        ("¿Qué hago para los sofocos?", "es"),
        ("tengo el ciclo irregular", "es"),
        ("fatiga y menopausia", "es"),
        ("no puedo dormir bien desde hace meses", "es"),
        ("me siento triste e irritable antes de la regla", "es"),
        ("¿la menopausia afecta a la memoria?", "es"),
        ("dolor pélvico durante la regla", "es"),
        ("sleep disorders menopause", "en"),
        ("ansiedad menopausia tratamiento", "es"),
        ("cognitive dysfunction menopause", "en"),
        ("¿qué es el síndrome premenstrual?", "es"),
    ]

    citations_count = 0
    for query, lang in test_queries:
        print(f"\n{'='*70}\nQuery: '{query}'\n{'='*70}")
        res = rag.search(query, lang, top_k=3)
        print(f"Mode: {res['search_mode']} | "
              f"Results: {res['results_count']} | "
              f"Time: {res['search_time']:.3f}s | "
              f"Top score: {res['top_score']:.3f}")

        for i, paper in enumerate(res["results"], 1):
            print(f"\n  [{i}] score={paper['relevance_score']:.3f} "
                  f"(sem={paper['semantic_score']:.3f}, "
                  f"kw={paper['keyword_score']:.3f}) "
                  f"conf={paper['confidence']}")
            print(f"      {paper['authors_str']} ({paper['year']})")
            print(f"      {paper['title'][:70]}...")

        if res["citations"]:
            citations_count += 1

    print("\n" + "=" * 70)
    print(f"✅ TEST COMPLETE — {citations_count}/{len(test_queries)} queries got citations")
    print("=" * 70)