import os
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Alura Agente - RAG", page_icon="🤖")

st.title("🤖 Alura Agente: Consultas de Ingeniería")
st.caption("Asistente virtual RAG impulsado por Llama 3 (vía Groq) para responder dudas técnicas.")

# Obtener API Key de Groq
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.warning("⚠️ No se encontró GROQ_API_KEY en los Secrets de Streamlit.")

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
    resultados = vectorstore.similarity_search(pregunta, k=2)
    contexto = "\n\n".join([doc.page_content for doc in resultados])

    if api_key:
        with st.spinner("Procesando respuesta con Llama 3..."):
            try:
                client = Groq(api_key=api_key)
                
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": "Eres un asistente técnico de ingeniería de Santos Pegasus Soluciones. Responde de manera precisa, clara y profesional utilizando ÚNICAMENTE la información dada en el contexto."
                        },
                        {
                            "role": "user",
                            "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"
                        }
                    ],
                    temperature=0.2
                )

                st.subheader("🤖 Respuesta del Agente:")
                st.write(response.choices[0].message.content)

                with st.expander("🔍 Ver contexto de origen (RAG)"):
                    st.info(contexto)

            except Exception as e:
                st.error(f"Error al conectar con la IA: {e}")
                st.subheader("📌 Respuesta directa (Búsqueda RAG):")
                st.info(contexto)
    else:
        st.subheader("📌 Contexto encontrado (Búsqueda RAG):")
        st.info(contexto)