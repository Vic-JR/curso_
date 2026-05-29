# 📚 Agente RAG en Streamlit con Groq

Asistente inteligente capaz de responder preguntas basadas en documentos locales.

## 🚀 Ejecución rápida

> [!CAUTION]
> **Compatibilidad de Python**: Esta aplicación requiere **Python 3.11** o **3.12**. 
> Se ha detectado que versiones superiores (como 3.14 Alpha) causan errores críticos en los clientes HTTP de HuggingFace.

1. **Crear carpeta de documentos**:
   ```bash
   mkdir docs
   ```
2. **Crear un entorno virtual (Recomendado)**:
   ```bash
   python -m venv venv
   # Activar en Windows: .\venv\Scripts\activate
   ```
3. **Configurar API Key**:
   Crea un archivo `.env` con: `GROQ_API_KEY=tu_key_aqui`
4. **Instalar dependencias**:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. **Lanzar la app**:
   ```bash
   streamlit run app.py
   ```

## 📂 Estructura
- `docs/`: Coloca tus archivos .txt aquí.
- `rag.py`: Motor de búsqueda y conexión con Groq.
- `app.py`: Interfaz de usuario.

## 🛠 Tecnologías
- **Groq**: Inferencia de LLM ultra rápida.
- **FAISS**: Búsqueda vectorial eficiente.
- **Sentence-Transformers**: Embeddings locales.