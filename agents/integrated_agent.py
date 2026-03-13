"""
PROFESSIONAL VERSION v3.3 - CONVERSATION MEMORY INTEGRATED

FIXES v3.3 (over v3.2):
✅ FIX 1: ConversationMemory conectada al agente principal
✅ FIX 2: answer() acepta session_id para multi-usuario
✅ FIX 3: Preguntas de perfil ("resúmeme mi historial") respondidas con datos REALES
✅ FIX 4: Preguntas de seguimiento enriquecidas con contexto real de conversación
✅ FIX 5: Perfil de usuario (nombre, edad, condiciones) inyectado en cada prompt
✅ FIX 6: Memoria guarda cada interacción automáticamente
✅ MANTIENE todos los fixes de v3.2
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from logger import HealthChatLogger
from agents.risk_agent import RiskAssessmentAgent, RiskLevel
from agents.rag_agent import RAGAgent
from agents.knowledge_base import MEDICAL_KNOWLEDGE_BASE
from agents.response_normalizer import ResponseNormalizer
from agents.emergency_handler_agent import EmergencyHandler
from conversation_memory import ConversationMemory
import time
import re
import random

# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────
_agent_instance = None

def get_agent():
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = IntegratedHealthAgent()
    return _agent_instance


class IntegratedHealthAgent:
    """Professional Health Triage Agent v3.3"""

    def __init__(self):
        self.logger = HealthChatLogger()
        self.risk_agent = RiskAssessmentAgent()
        self.rag_agent = RAGAgent(MEDICAL_KNOWLEDGE_BASE)
        self.normalizer = ResponseNormalizer()

        # ✅ NEW v3.3: Conversation memory
        self.memory = ConversationMemory()

        self.llm = ChatOllama(
            model="mistral",
            temperature=0.2,
            num_predict=350,
            repeat_penalty=1.1
        )

        self._setup_varied_phrases()
        self._setup_templates()

    # ─────────────────────────────────────────
    # SETUP
    # ─────────────────────────────────────────

    def _setup_varied_phrases(self):
        self.closings = {
            'es': ["Cuídate.", "Espero que mejores pronto."],
            'en': ["Take care.", "Hope you feel better soon."],
            'fr': ["Prenez soin de vous.", "J'espère que vous irez mieux bientôt."],
            'ar': ["اعتن بنفسك.", "أتمنى لك الشفاء العاجل."]
        }

        self.vague_symptoms = {
            'es': [r'\bme\s+(?:encuentro|siento)\s+mal\b', r'\bno\s+me\s+encuentro\s+bien\b',
                   r'\bme\s+duele\b(?!\s+\w)', r'\btengo\s+dolor\b(?!\s+de)'],
            'en': [r"\bi\s+(?:don'?t\s+feel\s+well|feel\s+(?:bad|sick|unwell))\b",
                   r'\bi\s+have\s+pain\b(?!\s+in)', r'\bi\s+hurt\b'],
            'fr': [r'\bje\s+ne\s+me\s+sens\s+pas\s+bien\b', r"\bje\s+me\s+sens\s+mal\b",
                   r"\bj'?ai\s+mal\b(?!\s+(?:à|au|aux))"],
            'ar': [r'مش\s*مرتاح', r'تعبان', r'عندي\s*ألم']
        }

    def _setup_templates(self):

        # ── LOW RISK ──────────────────────────────────────────────────────────
        self.low_risk_template = PromptTemplate(
            input_variables=["question", "context", "lang_name"],
            template=(
                "Health assistant. Reply ONLY in {lang_name}.\n"
                "Context: {context}\n\n"
                "Examples of correct answers:\n"
                "---\n"
                "Q: ¿Cómo puedo dormir mejor?\n"
                "A: Mantener un horario regular y evitar pantallas antes de dormir suele ayudar mucho. "
                "El estrés y la cafeína también afectan el descanso. "
                "Si el insomnio persiste más de dos semanas, vale la pena comentarlo con un médico.\n"
                "---\n"
                "Q: What causes frequent headaches?\n"
                "A: Frequent headaches can be related to stress, dehydration, or poor sleep. "
                "Keeping track of when they occur helps identify triggers. "
                "If they happen more than twice a week, it is worth discussing with a doctor.\n"
                "---\n"
                "Q: Tengo una verruga en el pie\n"
                "A: Las verrugas en el pie suelen estar causadas por un virus y no son peligrosas. "
                "Hay tratamientos de farmacia que funcionan bien. "
                "Si molesta mucho o no mejora, un dermatólogo puede valorarla.\n"
                "---\n"
                "Q: Mi hijo tiene 37.5 de fiebre\n"
                "A: Una temperatura de 37.5°C es una febrícula leve, no una fiebre alta. "
                "Es normal que aparezca con infecciones leves o después de ejercicio. "
                "Vigila si sube por encima de 38°C o si el niño parece muy decaído, en ese caso consulta al médico.\n"
                "---\n"
                "Same style — no greeting, no label at end:\n"
                "Q: {question}\n"
                "A:"
            )
        )

        # ── MEDIUM RISK ───────────────────────────────────────────────────────
        self.medium_risk_template = PromptTemplate(
            input_variables=["question", "context", "lang_name"],
            template=(
                "Health assistant. Reply ONLY in {lang_name}.\n"
                "Context: {context}\n\n"
                "Examples of correct answers:\n"
                "---\n"
                "Q: Llevo 3 días con dolor de garganta\n"
                "A: El dolor de garganta persistente puede deberse a una infección viral o bacteriana. "
                "Si va acompañado de fiebre o dificultad para tragar, conviene visitar al médico esta semana.\n"
                "---\n"
                "Q: I have had a dry cough for 2 weeks\n"
                "A: A cough lasting two weeks may be caused by a lingering infection, allergies, or reflux. "
                "It is worth seeing a doctor this week to rule out anything that needs treatment.\n"
                "---\n"
                "Q: Me arde al orinar desde ayer\n"
                "A: El ardor al orinar puede indicar una infección urinaria, que es tratable pero conviene confirmar. "
                "Consulta con un médico esta semana, o antes si aparece fiebre o el dolor empeora.\n"
                "---\n"
                "Same style — no greeting, no label at end:\n"
                "Q: {question}\n"
                "A:"
            )
        )

        # ── HIGH RISK ─────────────────────────────────────────────────────────
        self.high_risk_template = PromptTemplate(
            input_variables=["question", "context", "lang_name"],
            template=(
                "Health assistant. Reply ONLY in {lang_name}.\n"
                "Context: {context}\n\n"
                "Do NOT name specific diseases. Say symptoms 'may indicate a serious condition'.\n\n"
                "Examples of correct answers:\n"
                "---\n"
                "Q: He vomitado sangre\n"
                "A: Vomitar sangre puede indicar una causa grave que requiere evaluación urgente. "
                "Es importante acudir a urgencias hoy mismo sin esperar.\n"
                "---\n"
                "Q: Mi hijo tiene 39 de fiebre\n"
                "A: Una fiebre de 39°C en un niño requiere evaluación médica hoy. "
                "Si está muy decaído o la fiebre sigue subiendo, acude a urgencias sin esperar.\n"
                "---\n"
                "Q: I lost 10 pounds without dieting\n"
                "A: Unintentional weight loss can indicate an underlying condition that needs investigation. "
                "It is important to see a doctor soon for a proper evaluation.\n"
                "---\n"
                "Same style — no greeting, no phone numbers, no label at end:\n"
                "Q: {question}\n"
                "A:"
            )
        )

        # ── MENTAL HEALTH ─────────────────────────────────────────────────────
        self.mental_health_template = PromptTemplate(
            input_variables=["question", "lang_name"],
            template=(
                "Compassionate health assistant. Reply ONLY in {lang_name}.\n"
                "Match the SEVERITY of the response to the severity of what the person describes.\n\n"
                "Examples of correct answers:\n"
                "---\n"
                "Q: Estoy un poco triste hoy\n"
                "A: Es normal tener días en los que uno se siente bajo de ánimo. "
                "A veces un paseo, hablar con alguien cercano o simplemente descansar puede ayudar. "
                "Si la tristeza persiste varios días, no dudes en comentarlo con alguien de confianza.\n"
                "---\n"
                "Q: I'm feeling a bit down today\n"
                "A: It's normal to have off days. Sometimes getting some fresh air, "
                "connecting with someone you trust, or resting can help. "
                "If it continues for a few days, it may be worth talking to someone.\n"
                "---\n"
                "Q: Me siento muy triste y no veo sentido a nada\n"
                "A: Lo que describes suena muy pesado, y tiene todo el sentido buscar apoyo. "
                "Hablar con un profesional de salud mental puede ayudarte a entender lo que sientes "
                "y encontrar el camino para estar mejor. No tienes que pasar por esto solo/a.\n"
                "---\n"
                "Q: I feel very sad, I don't see the point anymore\n"
                "A: What you are describing sounds really hard, and it makes sense to reach out for support. "
                "Speaking with a mental health professional can truly make a difference. "
                "You don't have to go through this alone.\n"
                "---\n"
                "Q: Je me sens très triste depuis des semaines\n"
                "A: Ce que vous décrivez semble vraiment difficile à porter. "
                "Parler avec un professionnel de santé mentale peut vous aider à comprendre ce que vous ressentez "
                "et à trouver un chemin vers le mieux-être.\n"
                "---\n"
                "Q: Estoy deprimida desde hace semanas y no tengo ganas de nada\n"
                "A: Llevar semanas sintiéndote así es agotador, y merece atención. "
                "Te animo a hablar con tu médico o un psicólogo esta semana — "
                "hay formas de ayudarte a sentirte mejor y no tienes que enfrentarlo sola.\n"
                "---\n"
                "Same style — empathetic, proportional to severity, no diet advice, no label:\n"
                "Q: {question}\n"
                "A:"
            )
        )

        # ── CLARIFICATION ─────────────────────────────────────────────────────
        self.clarification_template = PromptTemplate(
            input_variables=["question", "lang_name"],
            template=(
                "Health assistant. Reply ONLY in {lang_name}.\n\n"
                "Examples:\n"
                "---\n"
                "Q: Me encuentro mal\n"
                "A: ¿Puedes contarme qué síntomas tienes exactamente y desde cuándo?\n"
                "---\n"
                "Q: I don't feel well\n"
                "A: Can you tell me what symptoms you have and how long they have been going on?\n"
                "---\n"
                "Same style — max 2 questions, no greeting, no label:\n"
                "Q: {question}\n"
                "A:"
            )
        )

    # ─────────────────────────────────────────
    # LANGUAGE DETECTION
    # ─────────────────────────────────────────

    def detect_language(self, text: str) -> dict:
        text_lower = text.lower()

        if len(re.findall(r'[\u0600-\u06FF]', text)) > 2:
            return {"name": "Arabic", "code": "ar", "is_greeting": False}

        greeting_patterns = {
            'es': [r'^hola\s*[!.?]?$', r'^buenos\s+días\s*[!.?]?$', r'^buenas\s*[!.?]?$'],
            'en': [r'^h(?:ello|i|ey)\s*[!.?]?$', r'^good\s+morning\s*[!.?]?$'],
            'fr': [r'^(?:bonjour|salut|bonsoir)\s*[!.?]?$'],
            'ar': [r'^(?:مرحبا|السلام|أهلا)\s*[!.?]?$']
        }
        for lang_code, patterns in greeting_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text_lower):
                    lang_names = {'es': 'español', 'en': 'English', 'fr': 'français', 'ar': 'Arabic'}
                    return {"name": lang_names[lang_code], "code": lang_code, "is_greeting": True}

        lang_scores = {"es": 0, "en": 0, "fr": 0}
        score_patterns = {
            "es": [r"\b(el|la|los|las|un|una|tengo|me|mi|dolor|fiebre|desde|hace|qué|cómo|por|llevo|siento|duele|estoy)\b"],
            "en": [r"\b(i|my|have|pain|fever|since|feel|what|how|why|the|is|are|been|hurt|ache|feeling)\b"],
            "fr": [r"\b(je|j'ai|mon|ma|mes|douleur|fièvre|depuis|quoi|comment|pourquoi|suis|fait|mal)\b"]
        }
        for lang, regex_list in score_patterns.items():
            for pattern in regex_list:
                lang_scores[lang] += len(re.findall(pattern, text_lower))

        if re.search(r'\b(tengo|desde hace|me duele|llevo)\b', text_lower):
            lang_scores["es"] += 3
        if re.search(r"\b(i've|i'm|i feel|i have)\b", text_lower):
            lang_scores["en"] += 3
        if re.search(r"\b(j'ai|depuis|je me|je suis)\b", text_lower):
            lang_scores["fr"] += 3

        detected_lang = max(lang_scores, key=lang_scores.get)
        if all(s == 0 for s in lang_scores.values()):
            detected_lang = "es"

        lang_names = {"es": "español", "en": "English", "fr": "français"}
        return {"name": lang_names[detected_lang], "code": detected_lang, "is_greeting": False}

    # ─────────────────────────────────────────
    # GREETING
    # ─────────────────────────────────────────

    def get_greeting_response(self, lang_code: str) -> str:
        greetings = {
            "es": "¡Hola! 👋 Soy tu asistente de salud. Puedo ayudarte con síntomas y orientarte sobre cuándo ir al médico. ¿En qué puedo ayudarte?",
            "en": "Hello! 👋 I'm your health assistant. I can help with symptoms and guide you on when to see a doctor. How can I help you?",
            "fr": "Bonjour! 👋 Je suis votre assistant santé. Je peux vous aider avec vos symptômes. Comment puis-je vous aider?",
            "ar": "مرحبا! 👋 أنا مساعدك الصحي. كيف يمكنني مساعدتك اليوم؟"
        }
        return greetings.get(lang_code, greetings["es"])

    # ─────────────────────────────────────────
    # VAGUE SYMPTOM
    # ─────────────────────────────────────────

    def _is_vague_symptom(self, question: str, lang_code: str) -> bool:
        if len(question.split()) > 8:
            return False
        q = question.lower()
        return any(re.search(p, q) for p in self.vague_symptoms.get(lang_code, []))

    # ─────────────────────────────────────────
    # CONTEXT OVERRIDES (v3.2)
    # ─────────────────────────────────────────

    def _context_override(self, question: str, lang_code: str):
        q = question.lower()

        toilet_blood_patterns = [
            r'sangre\s+(en\s+el\s+papel|al\s+limpiar|en\s+el\s+papel\s+higiénico)',
            r'blood\s+(on\s+the\s+paper|when\s+i\s+wipe|on\s+toilet\s+paper)',
            r'sang\s+(sur\s+le\s+papier|en\s+me\s+essuyant)',
            r'(papel|toilet\s+paper|papier\s+toilette).{0,20}(sangre|blood|sang)',
            r'(limpiarme|wipe|essuyer).{0,20}(sangre|blood|sang)',
        ]
        if any(re.search(p, q) for p in toilet_blood_patterns):
            return RiskLevel.LOW, False

        low_fever_child_patterns = [
            r'(niño|hijo|bebé|kid|child|baby|enfant).{0,40}(37[\.,][0-9])',
            r'(37[\.,][0-9]).{0,40}(niño|hijo|bebé|kid|child|baby|enfant)',
            r'(fiebre|fever|fièvre|temperatura|temperature).{0,20}(37[\.,][0-9])',
            r'(37[\.,][0-9]).{0,20}(grados|degrees|°|fiebre|fever)',
        ]
        if any(re.search(p, q) for p in low_fever_child_patterns):
            return RiskLevel.LOW, False

        mild_sadness_patterns = [
            r'(estoy|me\s+siento|me\s+encuentro)\s+(un\s+poco|algo|un\s+tanto)\s+(triste|bajo|mal)',
            r'(i\s+feel|i\'m|i\s+am)\s+(a\s+bit|kind\s+of|somewhat|a\s+little)\s+(sad|down|blue|low)',
            r'(je\s+me\s+sens|je\s+suis)\s+(un\s+peu|un\s+petit\s+peu)\s+(triste|déprimé)',
            r'(hoy|today|aujourd\'hui).{0,20}(triste|sad|triste)',
            r'(triste|sad|triste).{0,20}(hoy|today|aujourd\'hui)',
        ]
        if any(re.search(p, q) for p in mild_sadness_patterns):
            return RiskLevel.LOW, True

        return None

    # ─────────────────────────────────────────
    # ✅ NEW v3.3: MEMORY HELPERS
    # ─────────────────────────────────────────

    def _is_follow_up(self, question: str) -> bool:
        """Detect if question is a follow-up to previous context"""
        follow_up_patterns = [
            r'^(y|también|además|pero|entonces)\s',
            r'^(and|also|but|so|what about)\s',
            r'^(et|aussi|mais|alors)\s',
            r'(qué\s+más|what\s+else|quoi\s+d\'autre)',
            r'(en\s+ese\s+caso|in\s+that\s+case)',
            r'(me\s+dijiste|you\s+said|vous\s+avez\s+dit)',
            r'(sobre\s+eso|about\s+that|à\s+ce\s+sujet)',
            r'(antes\s+mencionaste|you\s+mentioned|vous\s+avez\s+mentionné)',
        ]
        q = question.lower().strip()
        if any(re.search(p, q, re.IGNORECASE) for p in follow_up_patterns):
            return True
        # Short questions with ? are likely follow-ups
        if len(question.split()) <= 5 and "?" in question:
            return True
        return False

    def _enrich_with_memory(self, question: str, session_id: str) -> str:
        """
        Enriches the question with real conversation context.

        RULES (to avoid context contamination):
        - Profile (name/age) only injected when question is personal/follow-up
        - Conditions only injected when DIRECTLY relevant to the question
        - Emergency/third-party questions (mi bebé, mi hijo) → NO profile injection
        - NEVER invents data
        """
        if not self.memory.has_context(session_id):
            return question

        q = question.lower()

        # ── RULE 1: Third-party questions → skip profile injection completely ──
        # "mi bebé", "mi hijo", "mi madre", etc. are about someone else
        third_party_patterns = [
            r'\b(mi\s+beb[eé]|mi\s+hij[oa]|mi\s+madr[e]|mi\s+padr[e]|mi\s+abuel[oa])\b',
            r'\b(my\s+baby|my\s+child|my\s+son|my\s+daughter|my\s+mother|my\s+father)\b',
            r'\b(mon\s+bébé|mon\s+enfant|ma\s+mère|mon\s+père)\b',
        ]
        if any(re.search(p, q) for p in third_party_patterns):
            return question

        # ── RULE 2: Emergency questions → skip profile injection ──────────────
        emergency_keywords = [
            r'\b(emergencia|urgencia|emergency|urgence)\b',
            r'\b(ataque|infarto|stroke|convulsi[oó]n)\b',
            r'\b(llama|llame|call|appelez)\s+(al\s+)?(112|911|médico|doctor)\b',
        ]
        if any(re.search(p, q) for p in emergency_keywords):
            return question

        profile = self.memory.extract_user_profile(session_id)

        # ── RULE 3: Follow-up question → inject recent context + profile ──────
        if self._is_follow_up(question):
            context = self.memory.get_context_for_prompt(session_id, max_messages=4)
            if context:
                return (
                    f"Conversación previa:\n{context}\n\n"
                    f"Pregunta de seguimiento: {question}\n"
                    f"Responde a la pregunta de seguimiento. "
                    f"No inventes información que el usuario no haya mencionado."
                )
            return question

        # ── RULE 4: Medication/allergy question → inject conditions if relevant
        medication_question = any(re.search(p, q) for p in [
            r'\b(medicamento|f[aá]rmaco|pastilla|medicine|drug|médicament)\b',
            r'\b(alergi[ao]|allergic?|allergie)\b',
            r'\b(tomar|take|prendre)\b',
            r'\b(interacci[oó]n|interaction)\b',
        ])

        if medication_question and 'conditions' in profile:
            conditions_str = ', '.join(profile['conditions'])
            name_str = f"El usuario se llama {profile['name']}. " if 'name' in profile else ""
            return (
                f"[{name_str}Condiciones médicas mencionadas por el usuario: {conditions_str}]\n"
                f"{question}"
            )

        # ── RULE 5: Personal health question → inject name + age only ─────────
        personal_question = any(re.search(p, q) for p in [
            r'\b(s[íi]ntoma|symptom|symptôme)\b',
            r'\b(me\s+duele|me\s+siento|i\s+feel|j[e\']?\s+me\s+sens)\b',
            r'\b(llevo|desde\s+hace|for\s+\d+\s+days?|depuis)\b',
        ])

        if personal_question and profile:
            profile_parts = []
            if 'name' in profile:
                profile_parts.append(f"El usuario se llama {profile['name']}")
            if 'age' in profile:
                profile_parts.append(f"tiene {profile['age']} años")
            if profile_parts:
                return f"[{', '.join(profile_parts)}]\n{question}"

        return question

    # ─────────────────────────────────────────
    # MAIN ANSWER — v3.3 with memory
    # ─────────────────────────────────────────

    def answer(self, question: str, session_id: str = "default") -> str:
        start_time = time.time()

        # STEP 0: EMERGENCY
        is_emergency, emergency_category = EmergencyHandler.is_emergency(question)
        if is_emergency:
            emergency_response = EmergencyHandler.get_emergency_response(emergency_category, question)
            lang_info = self.detect_language(question)
            self.memory.add_message(session_id, "user", question)
            self.memory.add_message(session_id, "assistant", emergency_response)
            self._log_interaction(
                question=question, answer=emergency_response, lang_info=lang_info,
                category=f"EMERGENCY_{emergency_category.upper()}",
                response_time=time.time() - start_time,
                risk_assessment={"risk_level": RiskLevel.HIGH}
            )
            return emergency_response

        # STEP 1: Language & Greeting
        lang_info = self.detect_language(question)
        if lang_info.get("is_greeting"):
            response = self.get_greeting_response(lang_info["code"])
            self.memory.add_message(session_id, "user", question)
            self.memory.add_message(session_id, "assistant", response)
            self._log_interaction(question=question, answer=response, lang_info=lang_info,
                                  category="greeting", response_time=time.time() - start_time)
            return response

        # STEP 1b: ✅ NEW — Profile/history question → answer from REAL memory
        if self.memory.is_profile_question(question):
            response = self.memory.build_profile_response(session_id, lang_info["code"])
            self.memory.add_message(session_id, "user", question)
            self.memory.add_message(session_id, "assistant", response)
            self._log_interaction(question=question, answer=response, lang_info=lang_info,
                                  category="profile_question", response_time=time.time() - start_time)
            return response

        # STEP 2: Vague → clarification
        if self._is_vague_symptom(question, lang_info["code"]):
            response = self._ask_clarification(question, lang_info)
            self.memory.add_message(session_id, "user", question)
            self.memory.add_message(session_id, "assistant", response)
            self._log_interaction(question=question, answer=response, lang_info=lang_info,
                                  category="clarification", response_time=time.time() - start_time)
            return response

        # STEP 3: Pre-safety check
        safety_override = self._pre_safety_check(question, lang_info["code"])
        if safety_override:
            response = self._final_cleanup(safety_override, is_emergency_response=False)
            self.memory.add_message(session_id, "user", question)
            self.memory.add_message(session_id, "assistant", response)
            self._log_interaction(question=question, answer=response, lang_info=lang_info,
                                  category="safety_override", response_time=time.time() - start_time)
            return response

        # STEP 4: Context override
        context_result = self._context_override(question, lang_info["code"])
        if context_result:
            risk_level, is_safety_sensitive = context_result
        else:
            risk_assessment = self.risk_agent.assess(question)
            risk_level = risk_assessment["risk_level"]
            is_safety_sensitive = risk_assessment.get("is_safety_sensitive", False)

        # STEP 5: RAG Context
        rag_data = self.rag_agent.search(question, lang_info["code"], top_k=2)
        context = (
            "\n".join([f"- {r['answer']}" for r in rag_data['results']])
            if rag_data['results'] else "General wellness and triage information."
        )

        # STEP 5b: ✅ NEW — Enrich question with real memory context
        enriched_question = self._enrich_with_memory(question, session_id)

        # STEP 6: Generate response
        response = self._generate_response(
            question=enriched_question, risk_level=risk_level,
            is_safety_sensitive=is_safety_sensitive,
            context=context, lang_info=lang_info
        )

        # STEP 7: Post-filter
        response = self._post_safety_filter(response, lang_info["code"])

        # STEP 8: Closing
        if not is_safety_sensitive:
            response = self._add_closing(response, risk_level, lang_info["code"])

        # STEP 9: Final cleanup
        response = self._final_cleanup(response, is_emergency_response=False)

        # STEP 10: ✅ NEW — Save to memory
        self.memory.add_message(session_id, "user", question)
        self.memory.add_message(session_id, "assistant", response)

        # STEP 11: Log
        self._log_interaction(
            question=question, answer=response, lang_info=lang_info,
            category=risk_level.value, response_time=time.time() - start_time,
            risk_assessment={"risk_level": risk_level, "is_safety_sensitive": is_safety_sensitive},
            rag_results=rag_data['results']
        )
        return response

    # ─────────────────────────────────────────
    # GENERATE
    # ─────────────────────────────────────────

    def _ask_clarification(self, question: str, lang_info: dict) -> str:
        chain = self.clarification_template | self.llm
        r = chain.invoke({"question": question, "lang_name": lang_info["name"]})
        return r.content if hasattr(r, 'content') else str(r)

    def _generate_response(self, question: str, risk_level: RiskLevel,
                           is_safety_sensitive: bool, context: str, lang_info: dict) -> str:
        if is_safety_sensitive:
            chain = self.mental_health_template | self.llm
            r = chain.invoke({"question": question, "lang_name": lang_info["name"]})
            return r.content if hasattr(r, 'content') else str(r)

        if risk_level == RiskLevel.LOW:
            chain = self.low_risk_template | self.llm
            r = chain.invoke({"question": question, "context": context, "lang_name": lang_info["name"]})
        elif risk_level == RiskLevel.MEDIUM:
            chain = self.medium_risk_template | self.llm
            r = chain.invoke({"question": question, "context": context, "lang_name": lang_info["name"]})
        else:
            chain = self.high_risk_template | self.llm
            r = chain.invoke({"question": question, "context": context, "lang_name": lang_info["name"]})

        return r.content if hasattr(r, 'content') else str(r)

    # ─────────────────────────────────────────
    # PRE-SAFETY CHECK (v3.2)
    # ─────────────────────────────────────────

    def _pre_safety_check(self, question: str, lang_code: str):
        q = question.lower()

        # ── LETHAL DOSE / SUICIDE METHOD — highest priority, treat as crisis ──
        if any(re.search(p, q) for p in [
            r'(pastillas?|pills?|medicamento|paracetamol|ibuprofeno?).{0,30}(morir|matar|suicid|die|kill|death|muerte)',
            r'(morir|matar|suicid|die|kill|death|muerte).{0,30}(pastillas?|pills?|medicamento|paracetamol|ibuprofeno?)',
            r'(cu[aá]ntas?.{0,10}pastillas?|how\s+many\s+pills?).{0,20}(morir|die|kill|muerte)',
            r'(dose|dosis|cantidad).{0,20}(letal|mortal|lethal|fatal)',
            r'(?:sobredosis|overdose|surdosage)',
        ]):
            return {
                'es': (
                    "💙 Lo que describes me preocupa y quiero que sepas que no estás solo/a.\n\n"
                    "Por favor, contacta ahora con los servicios de emergencia o una línea de apoyo en crisis. "
                    "Hay personas preparadas para escucharte y ayudarte en este momento."
                ),
                'en': (
                    "💙 What you're describing concerns me deeply, and I want you to know you're not alone.\n\n"
                    "Please reach out to emergency services or a crisis support line right now. "
                    "There are people ready to listen and help you through this."
                ),
                'fr': (
                    "💙 Ce que vous décrivez m'inquiète et je veux que vous sachiez que vous n'êtes pas seul(e).\n\n"
                    "Veuillez contacter les services d'urgence ou une ligne de crise maintenant. "
                    "Des personnes sont prêtes à vous écouter et à vous aider."
                ),
                'ar': (
                    "💙 ما تصفه يقلقني وأريدك أن تعرف أنك لست وحدك.\n\n"
                    "يرجى التواصل مع خدمات الطوارئ أو خط دعم الأزمات الآن. "
                    "هناك أشخاص مستعدون للاستماع إليك ومساعدتك."
                )
            }.get(lang_code, "💙 Please contact emergency services or a crisis support line now. You are not alone.")

        # ── NORMAL DOSAGE QUESTION (not lethal intent) ────────────────────────
        if any(re.search(p, q) for p in [
            r'\d+\s*(?:ibuprofeno?|paracetamol|aspirin\w*|pastillas?|pills?)',
            r'(?:puedo|can\s+i|puis-je).{0,20}(?:tomar|take|prendre).{0,20}\d+',
        ]):
            return {
                'es': "No puedo recomendar dosis específicas de medicamentos. Consulta con tu farmacéutico o médico. 💙",
                'en': "I cannot recommend specific medication doses. Please consult your pharmacist or doctor. 💙",
                'fr': "Je ne peux pas recommander des doses spécifiques. Consultez votre pharmacien ou médecin. 💙",
                'ar': "لا يمكنني التوصية بجرعات دواء محددة. يرجى استشارة طبيبك أو صيدلاني. 💙"
            }.get(lang_code, "I cannot recommend specific medication doses. Please consult a healthcare professional. 💙")

        if any(re.search(p, q) for p in [
            r'(suicid|self.?harm|automutil)',
            r'(quiero|want|veux)\s+(morir|die|mourir|matarme|hacerme\s+daño)',
            r'(no\s+quiero\s+vivir|don\'?t\s+want\s+to\s+live)',
            r'(quiero|voy\s+a)\s+hacerme\s+(daño|daño)',
        ]):
            return {
                'es': (
                    "💙 Lo que describes me preocupa y quiero que sepas que no estás solo/a.\n\n"
                    "Por favor, contacta ahora con los servicios de emergencia o una línea de apoyo en crisis. "
                    "Hay personas preparadas para escucharte y ayudarte en este momento."
                ),
                'en': (
                    "💙 What you're describing concerns me deeply, and I want you to know you're not alone.\n\n"
                    "Please reach out to emergency services or a crisis support line right now. "
                    "There are people ready to listen and help you through this."
                ),
                'fr': (
                    "💙 Ce que vous décrivez m'inquiète et je veux que vous sachiez que vous n'êtes pas seul(e).\n\n"
                    "Veuillez contacter les services d'urgence ou une ligne de crise maintenant. "
                    "Des personnes sont prêtes à vous écouter et à vous aider."
                ),
                'ar': (
                    "💙 ما تصفه يقلقني وأريدك أن تعرف أنك لست وحدك.\n\n"
                    "يرجى التواصل مع خدمات الطوارئ أو خط دعم الأزمات الآن. "
                    "هناك أشخاص مستعدون للاستماع إليك ومساعدتك."
                )
            }.get(lang_code, "💙 Please contact emergency services or a crisis support line now. You are not alone.")

        if any(re.search(p, q) for p in [
            r'(dime\s+qué\s+tengo|just\s+tell\s+me\s+what\s+i\s+have)',
            r'(no\s+me\s+mandes\s+al\s+médico|don\'?t\s+send\s+me\s+to\s+the\s+doctor)',
        ]):
            return {
                'es': "No puedo hacer diagnósticos, pero puedo orientarte sobre la urgencia de tus síntomas. ¿Qué síntomas tienes exactamente?",
                'en': "I can't make diagnoses, but I can guide you on urgency. What exactly are your symptoms?",
                'fr': "Je ne peux pas faire de diagnostics, mais je peux vous guider. Quels sont vos symptômes?",
                'ar': "لا يمكنني التشخيص، لكن يمكنني توجيهك. ما هي أعراضك بالضبط؟"
            }.get(lang_code, "I can't diagnose, but I can guide you. What are your symptoms?")

        return None

    # ─────────────────────────────────────────
    # POST-SAFETY FILTER
    # ─────────────────────────────────────────

    def _post_safety_filter(self, response: str, lang_code: str) -> str:
        for pattern, replacement in [
            (r'\byou have\b(?!\s+(?:the\s+right|a\s+point|enough))', 'you may have'),
            (r'\byou\'re suffering from\b', 'you may be experiencing'),
            (r'\btienes\b(?!\s+(?:razón|que|suerte|tiempo))', 'puedes tener'),
            (r'\bvous avez\b(?!\s+(?:besoin|raison))', 'vous pouvez avoir'),
            (r'\b(diagnosed with|diagnosticado con)\b', 'may indicate'),
        ]:
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)

        response = re.sub(r'Sentence\s+\d+:\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^\s*[-*]\s+', '', response, flags=re.MULTILINE)
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r'\s+', ' ', response).strip()

        return response

    # ─────────────────────────────────────────
    # CLOSING
    # ─────────────────────────────────────────

    def _add_closing(self, response: str, risk_level: RiskLevel, lang_code: str) -> str:
        if risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM):
            closing = random.choice(self.closings.get(lang_code, self.closings['es']))
            if not response.endswith(closing):
                response += f" {closing}"
        return response

    # ─────────────────────────────────────────
    # FINAL CLEANUP (v3.2)
    # ─────────────────────────────────────────

    def _final_cleanup(self, response: str, is_emergency_response: bool = False) -> str:
        for pat in [
            r'^estimado\s+paciente[,.]?\s*',
            r'^estimada?\s*[,.]?\s*',
            r'^dear\s+patient[,.]?\s*',
            r'^dear[,.]?\s*',
            r'^cher[,.]?\s*',
        ]:
            response = re.sub(pat, '', response, flags=re.IGNORECASE).strip()

        for pat in [
            r'\s*[🔴🟡🟢🟠]\s*(urgente|semi.urgente|no\s+urgente|clasificaci[oó]n)[^.]*\.?$',
            r'\s*(clasificaci[oó]n|classification)\s*:\s*[^\n.]+\.?$',
            r'\s*Urgente\s*[—\-]\s*busca\s+atenci[oó]n[^.]*\.?$',
            r'\s*Semi.urgente\s*[—\-][^.]*\.?$',
            r'\s*No\s+urgente[^.]*\.?$',
            r'\s*🔴[^.]*\.?$',
            r'\s*🟡[^.]*\.?$',
            r'\s*🟢[^.]*\.?$',
        ]:
            response = re.sub(pat, '', response, flags=re.IGNORECASE | re.MULTILINE).strip()

        if not is_emergency_response:
            response = re.sub(r'\b(112|911|15|3114|988|024|717\s*003\s*717)\b', '', response)

        response = re.sub(r'\s{2,}', ' ', response).strip()

        if response and response[-1] not in '.!?':
            response += '.'

        return response

    # ─────────────────────────────────────────
    # LOGGING
    # ─────────────────────────────────────────

    def _log_interaction(self, question: str, answer: str, lang_info: dict,
                         category: str, response_time: float,
                         risk_assessment: dict = None, rag_results: list = None,
                         quality_metrics: dict = None):
        metadata = {
            "model": "mistral",
            "lang_code": lang_info.get("code", "unknown"),
            "system_version": "v3.3",
        }
        if risk_assessment:
            metadata["risk_level"] = risk_assessment["risk_level"].value if hasattr(risk_assessment["risk_level"], 'value') else str(risk_assessment["risk_level"])
            metadata["is_safety_sensitive"] = risk_assessment.get("is_safety_sensitive", False)
        if rag_results:
            metadata["rag_used"] = True
        if quality_metrics:
            metadata["quality"] = quality_metrics

        self.logger.log_interaction(
            question=question, answer=answer,
            language=lang_info.get("name", "unknown"),
            category=category, response_time=response_time,
            metadata=metadata
        )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def health_answer(question: str, session_id: str = "default") -> str:
    return get_agent().answer(question, session_id=session_id)