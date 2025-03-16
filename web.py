import streamlit as st
import os
import json
import requests
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.storage import InMemoryStore
from langchain.llms import HuggingFaceHub

# GitHub raw URL for the resume PDF (Replace this with your actual URL)
GITHUB_PDF_URL = "Laiba_Muneer_resume.pdf"

# Local path to store the downloaded resume
PDF_FILE = "resume.pdf"

# Set Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load Hugging Face API Token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set. Please set it in your environment variables.")

# Download the resume from GitHub if it doesn't exist locally
if not os.path.exists(PDF_FILE):
    try:
        response = requests.get(GITHUB_PDF_URL)
        response.raise_for_status()
        with open(PDF_FILE, "wb") as f:
            f.write(response.content)
        st.sidebar.success("✅ Resume downloaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error downloading resume: {str(e)}")

# Load and process the PDF
if os.path.exists(PDF_FILE):
    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()
else:
    st.error("Resume file not found. Please check the GitHub URL.")

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text to embeddings
texts = [doc.page_content for doc in text_chunks]
embeddings = embedding_model.encode(texts, convert_to_tensor=False)
embedding_matrix = np.array(embeddings).astype("float32")

# Initialize FAISS index
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# Setup Retriever
vector_store = FAISS(
    embedding_function=embedding_model.encode,
    index=index,
    docstore=InMemoryStore(),
    index_to_docstore_id={str(i): str(i) for i in range(len(text_chunks))}
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initialize session state to store responses
if "responses" not in st.session_state:
    st.session_state.responses = []

# Streamlit UI
st.set_page_config(page_title="📜 AI Resume Chat", layout="wide")
st.title("🔍 Ask Questions About the Resume!")
st.markdown("This AI-powered chatbot lets you ask questions about the resume and fetches relevant answers.")

# User question input
question = st.text_input("Enter your question:")
if st.button("Ask"):
    if not question:
        st.warning("⚠️ Please enter a question.")
    else:
        retrieved_docs = retriever.get_relevant_documents(question)
        if retrieved_docs:
            st.success("✅ Here’s what we found:")
            for doc in retrieved_docs:
                st.markdown(f"**📄 Document:** {doc.page_content[:300]}...")

            # Store question and answer
            st.session_state.responses.append({
                "question": question,
                "answers": [doc.page_content[:300] for doc in retrieved_docs]
            })
        else:
            st.warning("❌ No relevant information found.")

# Display stored responses
if st.session_state.responses:
    st.write("### Stored Responses")
    st.json(st.session_state.responses)

# End button to download responses as JSON
if st.button("End"):
    if st.session_state.responses:
        json_str = json.dumps(st.session_state.responses, indent=2)
        st.download_button(
            label="📥 Download Responses as JSON",
            data=json_str,
            file_name="chatbot_responses.json",
            mime="application/json"
        )
    else:
        st.warning("⚠️ No responses to download yet.")
