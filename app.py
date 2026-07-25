import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Alura Agente - RAG", page_icon="🤖")

st.title("🤖 Alura Agente: Consultas de Ingeniería")
st.caption("Asistente virtual RAG para resolver dudas sobre políticas y arquitectura técnica.")

@st.cache_resource
def inicializar_vectorstore():
    loader = TextLoader("data/documento_santos_pegasus.txt", encoding="utf-8")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=0,
        separators=["\n\n", "\n", " "]
    )
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = inicializar_vectorstore()

# Inicializar variable de estado para la pregunta
if "query" not in st.session_state:
    st.session_state.query = ""

# Funciones para asignar la consulta desde los botones
def set_query(texto):
    st.session_state.query = texto

# Seccion de Preguntas Frecuentes (Botones rapidos)
st.markdown("💡 **Preguntas frecuentes:**")
col1, col2, col3 = st.columns(3)

with col1:
    st.button("🗄️ Base de Datos", on_click=set_query, args=("¿Cuál es la base de datos?",), use_container_width=True)
with col2:
    st.button("💻 Stack Back-End", on_click=set_query, args=("¿Cuál es el stack backend y lenguaje?",), use_container_width=True)
with col3:
    st.button("✉️ Contacto Soporte", on_click=set_query, args=("¿Cuál es el correo de soporte?",), use_container_width=True)

st.write("") # Espaciador

# Campo de entrada de texto enlazado a la variable de estado
pregunta = st.text_input("Escribe tu pregunta sobre la guía de ingeniería:", key="query")

# Procesar la búsqueda
if pregunta:
    resultados = vectorstore.similarity_search(pregunta, k=1)
    
    st.subheader("📌 Información relevante encontrada:")
    for i, doc in enumerate(resultados, 1):
        with st.container(border=True):
            st.markdown(f"**Resultado {i}:**")
            st.write(doc.page_content)