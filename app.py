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
    
    # Separar específicamente por saltos de línea doble para aislar secciones
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

pregunta = st.text_input("Escribe tu pregunta sobre la guía de ingeniería:")

if pregunta:
    # k=1 obliga al modelo a traer ÚNICAMENTE la coincidencia más relevante
    resultados = vectorstore.similarity_search(pregunta, k=1)
    
    st.subheader("💡 Información relevante encontrada:")
    for i, doc in enumerate(resultados, 1):
        with st.container(border=True):
            st.markdown(f"**Resultado {i}:**")
            st.write(doc.page_content)