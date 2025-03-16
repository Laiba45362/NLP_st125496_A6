import streamlit as st
import os
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain.llms import HuggingFaceHub
from langchain.retrievers import BM25Retriever
from transformers import pipeline

# Set up environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load Hugging Face API Token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set. Please set it in your environment variables.")

# Initialize Hugging Face LLM
hf_llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    huggingfacehub_api_token=hf_token,
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Define FAISS index file path
INDEX_PATH = "faiss_index.bin"

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI setup
st.set_page_config(page_title="📜 AI Document Search", layout="wide")
st.markdown("""
    <style>
        body { background-color: #f4f4f4; }
        .stButton > button { background-color: #4CAF50; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 AI-Powered Document Search")
st.markdown("Find answers in your documents with AI!")

# Sidebar for settings
st.sidebar.title("⚙️ Settings")
st.sidebar.header("📂 Upload PDF Documents")

# File uploader
uploaded_files = st.sidebar.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    pdf_files = [uploaded_file.name for uploaded_file in uploaded_files]
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
else:
    pdf_files = []

# Load documents
documents = []
for pdf_file in pdf_files:
    if os.path.exists(pdf_file):
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)

# Convert text to embeddings
texts = [doc.page_content for doc in text_chunks]
embeddings = embedding_model.encode(texts, convert_to_tensor=False)
embedding_matrix = np.array(embeddings).astype("float32")

# Initialize FAISS index
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    st.sidebar.success("✅ FAISS index loaded from disk.")
else:
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    faiss.write_index(index, INDEX_PATH)
    st.sidebar.success("✅ FAISS index created.")

# Setup Retriever
vector_store = FAISS(
    embedding_function=embedding_model.encode,
    index=index,
    docstore=InMemoryStore(),
    index_to_docstore_id={str(i): str(i) for i in range(len(text_chunks))}
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Question Answering UI
st.header("💡 Ask a Question")
question = st.text_input("Enter your question:")
if question:
    retrieved_docs = retriever.get_relevant_documents(question)
    if retrieved_docs:
        st.success("✅ Here’s what we found:")
        for doc in retrieved_docs:
            st.markdown(f"**📄 Document:** {doc.page_content[:300]}...")
    else:
        st.warning("❌ No relevant information found.")

# Footer
st.markdown("---")
st.markdown("💡 Created by [st125496](https://github.com/Laiba45362) ✨")
