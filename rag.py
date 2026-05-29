import os
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

class RAGAgent:
    def __init__(self, groq_api_key, embed_model):
        self.client = Groq(api_key=groq_api_key)
        self.embed_model = embed_model
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        self.index = None
        self.chunks = []

    def load_documents(self, directory):
        """Lee archivos de texto y PDF en el directorio docs/."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            return "Directorio creado. Por favor añade documentos."

        new_chunks = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename.endswith(".txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Limpiar espacios en blanco básicos
                        text = f.read().strip()
                        if text.strip():
                            new_chunks.extend(self.text_splitter.split_text(text))
                except Exception as e:
                    print(f"Error cargando TXT {filename}: {e}")
            elif filename.endswith(".pdf"):
                try:
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        content = page.extract_text()
                        if content: text += content + "\n"
                    if text.strip():
                        new_chunks.extend(self.text_splitter.split_text(text))
                except Exception as e:
                    print(f"Error cargando PDF {filename}: {e}")
        
        if not new_chunks:
            return "No se encontraron documentos válidos o contenido extraíble."
        
        self.chunks = new_chunks
        # Generación de embeddings e índice FAISS
        embeddings = self.embed_model.encode(self.chunks, show_progress_bar=False)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        return f"Éxito: {len(self.chunks)} fragmentos indexados."

    def get_context(self, query, k=3):
        """Busca los fragmentos más relevantes."""
        if self.index is None:
            return ""
        
        query_embedding = self.embed_model.encode([query], show_progress_bar=False)
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        relevant_chunks = [self.chunks[i] for i in indices[0] if i != -1]
        return "\n---\n".join(relevant_chunks)

    def query(self, user_question):
        """Genera respuesta usando Groq con el contexto recuperado."""
        context = self.get_context(user_question)
        
        prompt = f"""
        Eres un asistente experto. Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado.
        Si la respuesta no está en el contexto, di que no tienes suficiente información.

        Contexto:
        {context}

        Pregunta: 
        {user_question}
        """

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Asistente útil basado en documentos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            if "organization_restricted" in str(e):
                return "❌ Error: Tu cuenta de Groq ha sido restringida. Revisa tu facturación o límites en: https://console.groq.com/settings/billing"
            return f"❌ Error de Groq: {str(e)}"