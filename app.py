import streamlit as st
import os
from dotenv import load_dotenv
from rag import RAGAgent

load_dotenv()

st.set_page_config(page_title="Groq RAG Agent", page_icon="📑")

st.title("📑 Agente RAG con Groq")
st.sidebar.header("Configuración")

# Permitir usar la del .env o sobreescribirla manualmente
env_key = os.getenv("GROQ_API_KEY", "")
api_key = st.sidebar.text_input(
    "Groq API Key:", 
    value=env_key, 
    type="password", 
    help="Si la clave del .env falla, pega una nueva aquí.")

@st.cache_resource
def load_embed_model():
    """Carga el modelo de embeddings una sola vez."""
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        # Si falla por el cliente cerrado en Python 3.14, intentamos forzar carga local
        try:
            return SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
        except Exception:
            st.error("Error crítico de red/hilos. Python 3.14 es inestable para esta aplicación.")
            st.info("Se recomienda encarecidamente usar Python 3.11 o 3.12.")
            raise e

@st.cache_resource
def get_agent(key, _model):
    """Inicializa el agente usando el modelo cargado."""
    return RAGAgent(key, _model)

if api_key:
    embed_model = load_embed_model()
    # Inicializar el agente en la sesión
    st.session_state.agent = get_agent(api_key, embed_model)
    if "indexed" not in st.session_state:
        st.session_state.indexed = False

    # Botón para indexar
    if st.sidebar.button("Indexar Documentos (/docs)"):
        with st.spinner("Procesando documentos..."):
            result = st.session_state.agent.load_documents("./docs")
            if result.startswith("Éxito"):
                st.sidebar.success(result)
                st.session_state.indexed = True
            else:
                st.sidebar.error(result)

    # Historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Input de chat
    if prompt := st.chat_input("¿Qué deseas consultar?"):
        if not st.session_state.indexed:
            st.error("Por favor, indexa los documentos primero en la barra lateral.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.spinner("Groq está pensando..."):
                response = st.session_state.agent.query(prompt)
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
else:
    st.warning("Por favor, configura tu GROQ_API_KEY para continuar.")

st.sidebar.markdown("---")
st.sidebar.caption("Desarrollado con Groq + FAISS + Streamlit")