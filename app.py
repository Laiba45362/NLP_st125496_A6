import streamlit as st
import os
import numpy as np

# Attempt to import FAISS
try:
    import faiss
    from langchain_community.vectorstores import FAISS
    faiss_available = True
except ImportError:
    faiss_available = False

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain.retrievers import BM25Retriever
from transformers import pipeline

# Set Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if hf_token is None:
    st.error("🚨 Hugging Face API Token is missing! Set 'HUGGINGFACEHUB_API_TOKEN' in environment variables.")
    st.stop()

# Initialize Hugging Face LLM
hf_llm = pipeline("text2text-generation", model="google/flan-t5-large")

# Define FAISS index file path
INDEX_PATH = "faiss_index.bin"

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Streamlit UI ----
st.set_page_config(
    page_title="📚 AI Document Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI-Powered Document Search")
st.sidebar.title("⚙️ Settings")

# Theme Selection
theme = st.sidebar.radio("🌙 Choose Theme", ["Light", "Dark"], index=0)
st.markdown(f"""
    <style>
        body {{ background-color: {'#121212' if theme == 'Dark' else '#f5f5f5'}; color: {'white' if theme == 'Dark' else 'black'}; }}
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("📂 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

# Document processing
pdf_files = []
if uploaded_files:
    st.sidebar.success("📄 PDFs uploaded successfully!")
    pdf_files = [uploaded_file.name for uploaded_file in uploaded_files]
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

# Load and process documents
documents = []
if pdf_files:
    with st.spinner("🔄 Processing PDFs..."):
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                loader = PyPDFLoader(pdf_file)
                documents.extend(loader.load())

# Text splitting
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)

# Convert to embeddings
texts = [doc.page_content for doc in text_chunks]
embeddings = embedding_model.encode(texts, convert_to_tensor=False)
embedding_matrix = np.array(embeddings).astype("float32")

# Initialize FAISS
if faiss_available:
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        st.sidebar.success("✅ FAISS index loaded!")
    else:
        st.sidebar.warning("⚠️ FAISS index not found. Creating a new one...")
        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)
        faiss.write_index(index, INDEX_PATH)
        st.sidebar.success("✅ FAISS index saved!")

    # Create vector store
    docstore = InMemoryStore()
    index_to_docstore_id = {}

    document_objects = []
    for i, doc in enumerate(text_chunks):
        doc_object = Document(page_content=doc.page_content, metadata=doc.metadata)
        doc_object.metadata['id'] = str(i)
        document_objects.append(doc_object)
        index_to_docstore_id[str(i)] = str(i)

    docstore.mset([(str(i), doc) for i, doc in enumerate(document_objects)])

    vector_store = FAISS(
        embedding_function=embedding_model.encode,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
else:
    st.warning("⚠️ FAISS is not installed. Using basic keyword-based search instead.")
    retriever = BM25Retriever.from_documents(text_chunks) if text_chunks else None

# ---- User Interaction ----
st.header("💬 Ask a Question")
question = st.text_input("📝 Type your question here...")

if question:
    with st.spinner("🔍 Searching for answers..."):
        if retriever:
            retrieved_docs = retriever.get_relevant_documents(question)

            if not retrieved_docs:
                st.warning("⚠️ No relevant information found.")
            else:
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                answer = hf_llm(f"Answer based on these documents:\n{context}\n\nQuestion: {question}", max_length=512, do_sample=True)

                st.success("✅ Answer Generated!")
                st.markdown(f"### 🤖 AI Response:\n{answer[0]['generated_text']}")
                
                st.markdown("### 📚 Retrieved Documents:")
                for doc in retrieved_docs:
                    st.markdown(f"- **Content:** {doc.page_content[:200]}...  \n")
                    st.markdown(f"  **Metadata:** {doc.metadata}")

# ---- Footer ----
st.sidebar.markdown("---")
st.sidebar.markdown("🔹 Created by [st125496](https://github.com/Laiba45362)")

