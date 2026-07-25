import os
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Alura Agente - RAG Multi-PDF", page_icon="🤖", layout="wide")

st.title("🤖 Alura Agente: Consultas de Ingeniería")
st.caption("Asistente virtual RAG impulsado por Llama 3 (vía Groq) para responder dudas sobre documentación técnica en PDF.")

# Obtener API Key de Groq desde Secrets de Streamlit o variables de entorno local
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.warning("⚠️ No se encontró GROQ_API_KEY en los Secrets de Streamlit.")

@st.cache_resource
def inicializar_vectorstore():
    # PyPDFDirectoryLoader lee TODOS los archivos .pdf dentro de la carpeta 'data/'
    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    
    # Fragmentador ajustado para conservar contexto completo
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    docs = text_splitter.split_documents(documents)
    
    # Modelo de embeddings multilingüe en español para búsquedas precisas
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

with st.spinner("Indexando documentos PDF desde la carpeta data/..."):
    vectorstore = inicializar_vectorstore()

# Manejo de estado para preguntas frecuentes
if "query" not in st.session_state:
    st.session_state.query = ""

def set_query(texto):
    st.session_state.query = texto

st.markdown("💡 **Preguntas frecuentes:**")
col1, col2, col3 = st.columns(3)

with col1:
    st.button("🗄️ Base de Datos", on_click=set_query, args=("¿Cuál es la base de datos del proyecto?",), use_container_width=True)
with col2:
    st.button("💻 Stack Back-End", on_click=set_query, args=("¿Cuál es el stack backend y lenguaje principal?",), use_container_width=True)
with col3:
    st.button("✉️ Contacto Soporte", on_click=set_query, args=("¿Cuál es el correo de contacto para soporte?",), use_container_width=True)

st.write("")

pregunta = st.text_input("Escribe tu pregunta sobre la guía de ingeniería:", key="query")

if pregunta:
    # Búsqueda semántica con k=8 para rastrear múltiples PDFs
    resultados = vectorstore.similarity_search(pregunta, k=8)
    contexto = "\n\n---\n\n".join([doc.page_content for doc in resultados])

    if api_key:
        with st.spinner("Procesando respuesta con Llama 3.1..."):
            try:
                client = Groq(api_key=api_key)
                
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un asistente técnico de ingeniería de Santos Pegasus Soluciones. "
                                "Responde de manera precisa, clara, directa y profesional utilizando ÚNICAMENTE "
                                "la información extraída del contexto proporcionado de los documentos PDF. "
                                "Si la respuesta está en el contexto, indícala explícitamente sin hacer suposiciones."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Contexto extraído de los PDFs:\n{contexto}\n\nPregunta del usuario: {pregunta}"
                        }
                    ],
                    temperature=0.1
                )

                st.subheader("🤖 Respuesta del Agente:")
                st.write(response.choices[0].message.content)

                with st.expander("🔍 Ver fragmentos de contexto recuperados (RAG)"):
                    st.info(contexto)

            except Exception as e:
                st.error(f"Error al conectar con la API de Groq: {e}")
                st.subheader("📌 Fragmentos encontrados (Búsqueda RAG):")
                st.info(contexto)
    else:
        st.subheader("📌 Fragmentos encontrados (Búsqueda RAG):")
        st.info(contexto)