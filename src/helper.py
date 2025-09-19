import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

# load env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # still used for the LLM (optional)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# -------- Local embeddings wrapper (free, runs on your laptop) --------
class LocalEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

    def __call__(self, text):
        """Make this object callable, like expected by FAISS"""
        return self.embed_query(text)


# -------- PDF / chunking / vector store / chain functions --------
def get_pdf_text(pdf_docs):
    """Read list of uploaded PDF file-like objects and return combined text."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text


def get_text_chunks(text):
    """Split text into smaller chunks for embeddings/indexing."""
    # bigger chunks -> fewer embedding calls; tune as needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    """Create FAISS vector store from text chunks using local embeddings (free)."""
    embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
    # use embedding=<wrapper> (some versions use 'embedding', some 'embeddings')
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain(vector_store):
    """Return a conversational retrieval chain (LLM + retriever + memory)."""
    # Keep using Gemini for generation (optional). If you prefer fully offline,
    # you can swap this for a local LLM later.
    # More accurate, bigger model
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

# OR faster + free tier friendly
# llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)


    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conv_chain
