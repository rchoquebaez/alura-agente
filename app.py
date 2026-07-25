import os
import streamlit as st
from google import genai
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Alura Agente - RAG Gemini", page_icon="🤖")

st.title("🤖 Alura Agente: Consultas con Gemini")
st.caption("Asistente virtual RAG impulsado por Google Gemini para responder dudas técnicas.")

# Obtener API Key de Secrets o variable de entorno
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ No se encontró la API Key de Gemini. Configúrala en los Secrets de Streamlit.")

@st.cache_resource
def inicializar_vectorstore():
    loader = TextLoader("data/documento_santos_pegasus.txt", encoding="utf-8")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        separators=["\n\n", "\n", " "]
    )
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = inicializar_vectorstore()

# Manejo de estado para consultas rápidas
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
    # 1. Recuperar contexto con FAISS
    resultados = vectorstore.similarity_search(pregunta, k=2)
    contexto = "\n\n".join([doc.page_content for doc in resultados])

    if api_key:
        with st.spinner("Gemini está redactando la respuesta..."):
            try:
                client = genai.Client(api_key=api_key)
                prompt = f"""Eres un asistente técnico de ingeniería de Santos Pegasus Soluciones.
Responde de manera precisa, clara y profesional a la pregunta del usuario utilizando ÚNICAMENTE la siguiente información de contexto:

Contexto:
{contexto}

Pregunta del usuario:
{pregunta}

Respuesta:"""

                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=prompt
                )

                st.subheader("🤖 Respuesta del Agente:")
                st.write(response.text)

                with st.expander("🔍 Ver contexto de origen (RAG)"):
                    st.info(contexto)

            except Exception as e:
                st.error(f"Error al consultar Gemini: {e}")
    else:
        st.subheader("📌 Contexto encontrado (Modo Sin API Key):")
        st.write(contexto)