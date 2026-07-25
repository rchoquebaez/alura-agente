# 🤖 Alura Agente - Consultas de Ingeniería

Asistente virtual inteligente basado en una arquitectura **RAG (Retrieval-Augmented Generation)** diseñado para responder consultas técnicas y de arquitectura sobre la guía oficial de estándares de ingeniería de *Santos Pegasus Soluciones*.

---

## 🚀 Enlace de la Aplicación Desplegada

👉 **[Acceder a Alura Agente en vivo](https://alura-agente-rag.streamlit.app/)** 

---

## 📐 Arquitectura de la Solución

El agente implementa un flujo RAG optimizado para procesar documentos PDF y generar respuestas precisas sin alucinaciones:

### Componentes Clave:
1. **Multi-Document Loading:** Carga masiva e indexación de todos los archivos PDF presentes en la carpeta `data/` usando `PyPDFDirectoryLoader`.
2. **Text Splitting:** Fragmentación balanceada (`chunk_size=700`, `chunk_overlap=100`) para conservar la coherencia contextual de párrafos técnicos.
3. **Multilingual Vector Embeddings:** Generación de embeddings con el modelo multilingüe en español `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` e indexación vectorial en memoria con **FAISS**.
4. **Retrieval & Generation:** Búsqueda por similitud semántica amplia (`k=8`) integrada al LLM **Llama 3.1 8B Instant** a través de la infraestructura ultra-rápida de **Groq**.
5. **Frontend:** Interfaz web interactiva e intuitiva construida con **Streamlit**.

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.11+
* **Framework Web:** Streamlit
* **Orquestación RAG:** LangChain
* **Base de Datos Vectorial:** FAISS (Facebook AI Similarity Search)
* **Modelos de Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace)
* **LLM Inferencia:** Llama 3.1 8B Instant (vía Groq API)
* **Lector PDF:** PyPDF

---

## ❓ Ejemplos de Preguntas y Respuestas

### 1. Consulta sobre Base de Datos
* **Pregunta:** `¿Cuál es la base de datos del proyecto?`
* **Respuesta del Agente:** *"La base de datos relacional del proyecto es PostgreSQL 15, alojada en Oracle Cloud Infrastructure (OCI)."*

### 2. Consulta sobre Stack Back-End
* **Pregunta:** `¿Cuál es el stack backend y lenguaje principal?`
* **Respuesta del Agente:** *"El stack backend principal es Java 17+ con Spring Boot 3+, Spring Security y Spring Data JPA. El lenguaje principal es Java."*

### 3. Consulta sobre Soporte
* **Pregunta:** `¿Cuál es el correo de contacto para soporte?`
* **Respuesta del Agente:** *"El correo oficial de soporte técnico es soporte.dev@santospegasus.com."*

---

## 💻 Instrucciones para Ejecución Local

Para ejecutar el proyecto en tu máquina local:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/rchoquebaez/alura-agente.git](https://github.com/rchoquebaez/alura-agente.git)
   cd alura-agente