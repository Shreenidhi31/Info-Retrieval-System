# Info-Retrieval-System
Overview

This project is a Retrieval-Augmented Generation (RAG) system built with LangChain, FAISS, and Streamlit.
It allows users to upload PDF documents and ask questions. The system retrieves relevant text from the PDFs and generates answers using Google Gemini (Generative AI).

✅ Inspired by PaLM-based tutorials, but updated to Gemini API (PaLM is deprecated).
✅ Maintains conversation history using LangChain’s ConversationBufferMemory.
✅ Uses FAISS vector store for fast similarity search.

🛠️ Tech Stack

Python

Streamlit (UI)

LangChain (chains, memory, retrieval)

FAISS (vector database)

Google Generative AI (Gemini) – for LLM and embeddings

Sentence-Transformers (optional, local embeddings alternative)

⚙️ Features

📂 Upload multiple PDF files.

🔍 Extracts text and splits into chunks.

🧠 Stores embeddings in FAISS for similarity search.

💬 Chatbot answers based on retrieved context.

📝 Keeps full chat history during a session.

🔑 Setup & Installation
# Clone this repository
git clone https://github.com/your-username/info-retrieval-system.git
cd info-retrieval-system

# Create and activate virtual environment
python -m venv genai
source genai/Scripts/activate   # (Windows Git Bash)

# Install dependencies
pip install -r requirements.txt


Create a .env file in the project root:

GOOGLE_API_KEY=your_api_key_here


Run the app:

streamlit run app.py

📌 Notes

This project originally used PaLM, but since PaLM is deprecated, it now uses Gemini (ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings).

Free Gemini API keys may have 0 quota for embeddings/generation unless billing is enabled.

As an alternative, you can use Sentence-Transformers (local embeddings) to run fully free/offline.

📚 Project Flow (RAG)

PDF Upload → Extract text with PyPDF2.

Text Chunking → Split into smaller pieces.

Embeddings → Convert chunks into vectors (Gemini or Sentence-Transformers).

FAISS Vector Store → Store and retrieve similar chunks.

Conversational Chain → LLM (Gemini) generates answers with context.

Memory → Stores all previous Q&A for chat continuity.
