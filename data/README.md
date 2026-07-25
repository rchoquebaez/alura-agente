# 🤖 Alura Agente - Consultas de Ingeniería

Asistente virtual inteligente basado en una arquitectura **RAG (Retrieval-Augmented Generation)** diseñado para responder consultas técnicas y de arquitectura sobre la guía oficial de estándares de ingeniería de *Santos Pegasus Soluciones*.

---

## 🚀 Enlace de la Aplicación Desplegada

👉 **[Acceder a Alura Agente en vivo](https://alura-agente-rag.streamlit.app/)** 

---

## 📐 Arquitectura de la Solución

El agente implementa un flujo RAG optimizado para procesar documentos PDF y generar respuestas precisas sin alucinaciones:

### Componentes Clave:
1. **Document Loading:** Carga del documento PDF oficial desde la carpeta `data/` usando `PyPDFLoader`.
2. **Text Splitting:** Fragmentación del contenido (`chunk_size=500`, `chunk_overlap=50`) para asegurar coherencia contextual.
3. **Vector Embeddings & Storage:** Generación de embeddings con HuggingFace (`all-MiniLM-L6-v2`) e indexación vectorial en memoria con **FAISS**.
4. **Retrieval & Generation:** Recuperación por similitud semántica integrada al modelo **Llama 3.1 8B Instant** a través de la infraestructura de alta velocidad de **Groq**.
5. **Frontend:** Interfaz web interactiva e intuitiva construida con **Streamlit**.

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.11+
* **Framework Web:** Streamlit
* **Orquestación RAG:** LangChain
* **Base de Datos Vectorial:** FAISS (Facebook AI Similarity Search)
* **Modelos de Embeddings:** `all-MiniLM-L6-v2` (HuggingFace)
* **LLM Inferencia:** Llama 3.1 8B Instant (vía Groq API)
* **Lector PDF:** PyPDF

---

## ❓ Ejemplos de Preguntas y Respuestas

### 1. Consulta sobre Base de Datos
* **Pregunta:** `¿Cuál es la base de datos del proyecto?`
* **Respuesta del Agente:** *"La base de datos relacional del proyecto es PostgreSQL 15, alojada en Oracle Cloud Infrastructure (OCI)."*

### 2. Consulta sobre Arquitectura
* **Pregunta:** `¿Cuál es el stack backend y lenguaje principal?`
* **Respuesta del Agente:** *"El backend está desarrollado en Python 3.11+ utilizando FastAPI para servicios RESTful y utiliza Arquitectura Hexagonal."*

### 3. Consulta sobre Soporte
* **Pregunta:** `¿Cuál es el correo de contacto para soporte?`
* **Respuesta del Agente:** *"El correo oficial de soporte técnico es soporte.dev@santospegasus.com."*

---

## 💻 Instrucciones para Ejecución Local

Para ejecutar el proyecto en tu máquina local:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/rchoquebaez/alura-agente.git
   cd alura-agente