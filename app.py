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
    # 1. Cargar el documento
    loader = TextLoader("data/documento_santos_pegasus.txt", encoding="utf-8")
    documents = loader.load()
    
    # 2. Dividir el texto en fragmentos pequeños (Chunks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,      # Tamaño máximo de caracteres por fragmento
        chunk_overlap=30     # Traslape entre fragmentos para no perder contexto
    )
    docs = text_splitter.split_documents(documents)
    
    # 3. Crear embeddings y vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Cargar base vectorial
vectorstore = inicializar_vectorstore()

# Interfaz de usuario
pregunta = st.text_input("Escribe tu pregunta sobre la guía de ingeniería:")

if pregunta:
    # Buscar los 2 fragmentos más relevantes (k=2)
    resultados = vectorstore.similarity_search(pregunta, k=2)
    
    st.subheader("💡 Información relevante encontrada:")
    for i, doc in enumerate(resultados, 1):
        with st.container(border=True):
            st.markdown(f"**Resultado {i}:**")
            st.write(doc.page_content)