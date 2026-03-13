"""
Medical Knowledge Base
Contains reliable health information for RAG system
Multi-language support: Spanish, English, French, Arabic
"""

# Medical Knowledge Base - Structured data
MEDICAL_KNOWLEDGE_BASE = [
    # GENERAL HEALTH
    {
        "id": "gen_001",
        "category": "general_health",
        "topic": "hydration",
        "es": {
            "question": "¿Cuánta agua debo beber al día?",
            "answer": "Se recomienda beber entre 2 y 3 litros de agua al día para adultos. La cantidad exacta depende del peso corporal, nivel de actividad física y clima. Una buena señal de hidratación adecuada es la orina de color amarillo claro."
        },
        "en": {
            "question": "How much water should I drink daily?",
            "answer": "It's recommended to drink 2-3 liters of water per day for adults. The exact amount depends on body weight, physical activity level, and climate. A good sign of proper hydration is light yellow urine."
        },
        "fr": {
            "question": "Combien d'eau dois-je boire par jour?",
            "answer": "Il est recommandé de boire entre 2 et 3 litres d'eau par jour pour les adultes. La quantité exacte dépend du poids corporel, du niveau d'activité physique et du climat."
        }
    },
    
    # HEADACHES
    {
        "id": "symp_001",
        "category": "symptoms",
        "topic": "headache",
        "es": {
            "question": "¿Por qué me duele la cabeza?",
            "answer": "Los dolores de cabeza pueden tener múltiples causas: deshidratación (causa más común), falta de sueño, tensión muscular en cuello y hombros, estrés, exposición prolongada a pantallas, o cambios hormonales. Para aliviarlos: bebe agua, descansa en lugar oscuro, relaja cuello y hombros, y aplica compresas frías o calientes."
        },
        "en": {
            "question": "Why do I have a headache?",
            "answer": "Headaches can have multiple causes: dehydration (most common), lack of sleep, muscle tension in neck and shoulders, stress, prolonged screen exposure, or hormonal changes. To relieve them: drink water, rest in a dark place, relax neck and shoulders, and apply cold or hot compresses."
        },
        "fr": {
            "question": "Pourquoi ai-je mal à la tête?",
            "answer": "Les maux de tête peuvent avoir plusieurs causes: déshydratation (la plus courante), manque de sommeil, tension musculaire, stress, exposition prolongée aux écrans, ou changements hormonaux."
        }
    },
    
    # STRESS
    {
        "id": "prev_001",
        "category": "prevention",
        "topic": "stress_management",
        "es": {
            "question": "¿Cómo reducir el estrés?",
            "answer": "Técnicas efectivas para reducir el estrés: 1) Respiración profunda (inhala 4 seg, mantén 4 seg, exhala 4 seg), 2) Ejercicio regular (30 min al día), 3) Sueño adecuado (7-8 horas), 4) Meditación o mindfulness (10 min diarios), 5) Tiempo para hobbies y actividades placenteras, 6) Mantener conexiones sociales."
        },
        "en": {
            "question": "How to reduce stress?",
            "answer": "Effective stress reduction techniques: 1) Deep breathing (inhale 4 sec, hold 4 sec, exhale 4 sec), 2) Regular exercise (30 min daily), 3) Adequate sleep (7-8 hours), 4) Meditation or mindfulness (10 min daily), 5) Time for hobbies and pleasant activities, 6) Maintain social connections."
        },
        "fr": {
            "question": "Comment réduire le stress?",
            "answer": "Techniques efficaces pour réduire le stress: respiration profonde, exercice régulier, sommeil adéquat, méditation, temps pour les loisirs, et maintenir des connexions sociales."
        }
    },
    
    # FATIGUE
    {
        "id": "symp_002",
        "category": "symptoms",
        "topic": "fatigue",
        "es": {
            "question": "¿Por qué estoy siempre cansado?",
            "answer": "Las causas comunes de fatiga crónica incluyen: falta de sueño de calidad, dieta desequilibrada (deficiencia de hierro o vitamina B12), sedentarismo, deshidratación, estrés crónico, o problemas de tiroides. Recomendaciones: establece rutina de sueño regular, come alimentos nutritivos, haz ejercicio moderado, y mantente hidratado."
        },
        "en": {
            "question": "Why am I always tired?",
            "answer": "Common causes of chronic fatigue include: lack of quality sleep, unbalanced diet (iron or vitamin B12 deficiency), sedentary lifestyle, dehydration, chronic stress, or thyroid issues. Recommendations: establish regular sleep routine, eat nutritious foods, do moderate exercise, and stay hydrated."
        },
        "fr": {
            "question": "Pourquoi suis-je toujours fatigué?",
            "answer": "Les causes courantes de fatigue chronique incluent: manque de sommeil de qualité, alimentation déséquilibrée, sédentarité, déshydratation, stress chronique."
        }
    },
    
    # DIABETES
    {
        "id": "cond_001",
        "category": "conditions",
        "topic": "diabetes",
        "es": {
            "question": "¿Qué es la diabetes y cómo prevenirla?",
            "answer": "La diabetes tipo 2 es una condición donde el cuerpo no usa la insulina correctamente. Factores de riesgo: sobrepeso, sedentarismo, historial familiar. Prevención: mantén peso saludable, ejercicio regular (150 min/semana), dieta rica en fibra y baja en azúcares refinados, controla el estrés, y realiza chequeos médicos regulares."
        },
        "en": {
            "question": "What is diabetes and how to prevent it?",
            "answer": "Type 2 diabetes is a condition where the body doesn't use insulin correctly. Risk factors: overweight, sedentary lifestyle, family history. Prevention: maintain healthy weight, regular exercise (150 min/week), high-fiber low-refined-sugar diet, manage stress, and regular medical checkups."
        },
        "fr": {
            "question": "Qu'est-ce que le diabète et comment le prévenir?",
            "answer": "Le diabète de type 2 est une condition où le corps n'utilise pas correctement l'insuline. Prévention: maintenir un poids santé, exercice régulier, alimentation riche en fibres."
        }
    },
    
    # BLOOD PRESSURE
    {
        "id": "cond_002",
        "category": "conditions",
        "topic": "blood_pressure",
        "es": {
            "question": "¿Cómo controlar la presión arterial?",
            "answer": "Para mantener presión arterial saludable: reduce el consumo de sal (menos de 5g/día), mantén peso adecuado, ejercicio regular (caminar, nadar), limita el alcohol, maneja el estrés, duerme bien, y aumenta consumo de potasio (plátanos, espinacas). La presión normal es menor a 120/80 mmHg."
        },
        "en": {
            "question": "How to control blood pressure?",
            "answer": "To maintain healthy blood pressure: reduce salt intake (less than 5g/day), maintain proper weight, regular exercise (walking, swimming), limit alcohol, manage stress, sleep well, and increase potassium intake (bananas, spinach). Normal pressure is below 120/80 mmHg."
        },
        "fr": {
            "question": "Comment contrôler la tension artérielle?",
            "answer": "Pour maintenir une tension saine: réduire le sel, maintenir un poids approprié, exercice régulier, limiter l'alcool, gérer le stress."
        }
    },
    
    # SLEEP
    {
        "id": "gen_002",
        "category": "general_health",
        "topic": "sleep",
        "es": {
            "question": "¿Cómo dormir mejor?",
            "answer": "Consejos para mejorar el sueño: mantén horario regular (acuéstate y levántate a la misma hora), evita pantallas 1 hora antes de dormir, crea ambiente oscuro y fresco (18-20°C), evita cafeína después de las 14:00, haz ejercicio pero no cerca de la hora de dormir, y practica técnicas de relajación."
        },
        "en": {
            "question": "How to sleep better?",
            "answer": "Tips to improve sleep: maintain regular schedule (same bedtime and wake time), avoid screens 1 hour before bed, create dark and cool environment (18-20°C), avoid caffeine after 2 PM, exercise but not close to bedtime, and practice relaxation techniques."
        },
        "fr": {
            "question": "Comment mieux dormir?",
            "answer": "Conseils pour améliorer le sommeil: maintenir un horaire régulier, éviter les écrans avant de dormir, créer un environnement sombre et frais, éviter la caféine l'après-midi."
        }
    },
    
    # BODY PAIN
    {
        "id": "symp_003",
        "category": "symptoms",
        "topic": "body_pain",
        "es": {
            "question": "¿Por qué me duele todo el cuerpo?",
            "answer": "El dolor corporal generalizado puede deberse a: estrés y tensión muscular, falta de actividad física o ejercicio excesivo, mala postura, deshidratación, falta de sueño, o inicio de enfermedad viral. Ayuda: descanso adecuado, hidratación, estiramientos suaves, baño caliente, y técnicas de relajación."
        },
        "en": {
            "question": "Why does my whole body hurt?",
            "answer": "Generalized body pain can be due to: stress and muscle tension, lack of physical activity or excessive exercise, poor posture, dehydration, lack of sleep, or onset of viral illness. Helps: adequate rest, hydration, gentle stretching, warm bath, and relaxation techniques."
        },
        "fr": {
            "question": "Pourquoi tout mon corps me fait mal?",
            "answer": "La douleur corporelle généralisée peut être due au stress, manque d'activité, mauvaise posture, déshydratation, manque de sommeil."
        }
    },
    
    # EXERCISE
    {
        "id": "prev_002",
        "category": "prevention",
        "topic": "exercise",
        "es": {
            "question": "¿Qué ejercicio hacer en casa?",
            "answer": "Ejercicios efectivos en casa sin equipo: 1) Caminata en el sitio (10 min), 2) Sentadillas (3 series de 10), 3) Flexiones (adaptadas a tu nivel), 4) Plancha (30 seg-1 min), 5) Estiramientos (5-10 min). Objetivo: 30 minutos de actividad diaria. Comienza despacio y aumenta gradualmente."
        },
        "en": {
            "question": "What exercise to do at home?",
            "answer": "Effective home exercises without equipment: 1) Walking in place (10 min), 2) Squats (3 sets of 10), 3) Push-ups (adapted to your level), 4) Plank (30 sec-1 min), 5) Stretching (5-10 min). Goal: 30 minutes daily activity. Start slowly and increase gradually."
        },
        "fr": {
            "question": "Quel exercice faire à la maison?",
            "answer": "Exercices efficaces à domicile: marche sur place, squats, pompes adaptées, planche, étirements. Objectif: 30 minutes d'activité quotidienne."
        }
    },
    
    # NUTRITION
    {
        "id": "prev_003",
        "category": "prevention",
        "topic": "nutrition",
        "es": {
            "question": "¿Qué alimentos son saludables?",
            "answer": "Base de alimentación saludable: 1) Frutas y verduras variadas (5 porciones/día), 2) Cereales integrales (arroz, avena, pan integral), 3) Proteínas magras (pollo, pescado, legumbres), 4) Grasas saludables (aceite de oliva, aguacate, frutos secos), 5) Agua abundante. Limita: azúcares añadidos, sal excesiva, y alimentos ultraprocesados."
        },
        "en": {
            "question": "What foods are healthy?",
            "answer": "Healthy eating foundation: 1) Varied fruits and vegetables (5 servings/day), 2) Whole grains (rice, oats, whole bread), 3) Lean proteins (chicken, fish, legumes), 4) Healthy fats (olive oil, avocado, nuts), 5) Plenty of water. Limit: added sugars, excessive salt, and ultra-processed foods."
        },
        "fr": {
            "question": "Quels aliments sont sains?",
            "answer": "Base d'alimentation saine: fruits et légumes variés, céréales complètes, protéines maigres, graisses saines, eau abondante. Limiter: sucres ajoutés, sel excessif."
        }
    }
]

# Helper function to search knowledge base
def search_knowledge_base(query: str, language: str = "es", top_k: int = 3):
    """
    Simple search function (will be replaced by vector search in RAG agent)
    """
    query_lower = query.lower()
    results = []
    
    for entry in MEDICAL_KNOWLEDGE_BASE:
        if language in entry:
            question = entry[language].get("question", "").lower()
            answer = entry[language].get("answer", "")
            
            # Simple keyword matching
            words = query_lower.split()
            matches = sum(1 for word in words if word in question or word in answer.lower())
            
            if matches > 0:
                results.append({
                    "id": entry["id"],
                    "category": entry["category"],
                    "topic": entry["topic"],
                    "question": entry[language]["question"],
                    "answer": entry[language]["answer"],
                    "relevance_score": matches
                })
    
    # Sort by relevance and return top_k
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:top_k]


# Statistics about knowledge base
def get_kb_stats():
    """Get statistics about the knowledge base"""
    total_entries = len(MEDICAL_KNOWLEDGE_BASE)
    
    categories = {}
    topics = {}
    languages = set()
    
    for entry in MEDICAL_KNOWLEDGE_BASE:
        # Count categories
        cat = entry["category"]
        categories[cat] = categories.get(cat, 0) + 1
        
        # Count topics
        topic = entry["topic"]
        topics[topic] = topics.get(topic, 0) + 1
        
        # Count languages
        for key in entry.keys():
            if key not in ["id", "category", "topic"]:
                languages.add(key)
    
    return {
        "total_entries": total_entries,
        "categories": categories,
        "topics": list(topics.keys()),
        "languages": list(languages),
        "coverage": f"{total_entries} medical topics covered"
    }


if __name__ == "__main__":
    # Test search
    print("=" * 70)
    print("MEDICAL KNOWLEDGE BASE - STATISTICS")
    print("=" * 70)
    
    stats = get_kb_stats()
    print(f"\nTotal Entries: {stats['total_entries']}")
    print(f"Languages: {', '.join(stats['languages'])}")
    print(f"Categories: {stats['categories']}")
    print(f"\nTopics covered:")
    for topic in stats['topics']:
        print(f"  - {topic}")
    
    print("\n" + "=" * 70)
    print("TESTING SEARCH")
    print("=" * 70)
    
    # Test queries
    test_queries = [
        ("dolor de cabeza", "es"),
        ("how to reduce stress", "en"),
        ("todo me duele", "es"),
    ]
    
    for query, lang in test_queries:
        print(f"\nQuery: '{query}' (lang: {lang})")
        results = search_knowledge_base(query, lang, top_k=2)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"  Topic: {result['topic']}")
                print(f"  Q: {result['question']}")
                print(f"  A: {result['answer'][:100]}...")
        else:
            print("  No results found")
        
        print("-" * 70)