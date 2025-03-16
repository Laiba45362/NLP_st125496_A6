import streamlit as st
import os
import json
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.storage import InMemoryStore
from langchain.llms import HuggingFaceHub

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

# Initialize session state to store responses
if "responses" not in st.session_state:
    st.session_state.responses = []

# Question Answering UI
st.header("💡 Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Ask"):
    if not question:
        st.write("Please enter a question.")
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
            label="Download Responses as JSON",
            data=json_str,
            file_name="chatbot_responses.json",
            mime="application/json"
        )
    else:
        st.write("No responses to download yet.")
