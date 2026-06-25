# Agentes de Inteligencia Artificial para Aplicaciones Empresariales

> Sistema conversacional multiagente para información sobre salud hormonal femenina.  
> Trabajo Fin de Máster — Universidad Pablo de Olavide.

**Despliegue**: https://healthbot-ys0j.onrender.com

## Descripción

Las respuestas del sistema se generan con Llama 3.3 70B (servido a través de Groq) ancladas a un corpus de 2.527 artículos de PubMed mediante recuperación semántica. Cubre consultas sobre ciclo menstrual, perimenopausia, menopausia y función tiroidea, en español, inglés, francés y árabe.

## Requisitos

- Python 3.11
- Una clave de API de Groq (gratuita): https://console.groq.com/keys

## Instalación

```bash
git clone https://github.com/yasminekharbach64-cmd/tfm_multiagent.git
cd tfm_multiagent

python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows
# source venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
cp .env.example .env            # añadir GROQ_API_KEY en el fichero .env

python api.py
```

La aplicación queda accesible en http://localhost:5000. El primer arranque descarga el modelo de embeddings multilingüe (unos 120 MB), por lo que tarda unos segundos más.

## Docker

```bash
docker build -t healthbot .
docker run --env-file .env -p 5000:5000 healthbot
```