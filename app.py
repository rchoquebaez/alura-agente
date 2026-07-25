import os
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(
    page_title="Alura Agente - Santos Pegasus",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Alura Agente: Consultas de Ingeniería")
st.caption("Asistente virtual RAG para resolver dudas sobre políticas y arquitectura técnica.")

# 1. Cargar y procesar el documento con embeddings locales gratuitos
@st.cache_resource
def preparar_vectorstore():
    ruta_documento = os.path.join("data", "documento_santos_pegasus.txt")
    loader = TextLoader(ruta_documento, encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40
    )
    splits = text_splitter.split_documents(docs)
    
    # Modelo de embeddings gratuito de HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

with st.spinner("Cargando modelo de embeddings y base vectorial..."):
    vectorstore = preparar_vectorstore()

# 2. Interfaz y búsqueda RAG
query = st.text_input(
    "Escribe tu pregunta sobre la guía de ingeniería:",
    placeholder="Ej: ¿Cuál es el lenguaje principal para el desarrollo Back-end?"
)

if query:
    with st.spinner("Buscando respuesta en el documento..."):
        # Búsqueda semántica directa sobre la base vectorial
        docs_relacionados = vectorstore.similarity_search(query, k=2)
        
        st.subheader("💡 Información relevante encontrada:")
        if docs_relacionados:
            for i, doc in enumerate(docs_relacionados):
                st.info(f"**Resultado {i+1}:**\n\n{doc.page_content}")
        else:
            st.warning("No se encontró información que coincida en el documento.")