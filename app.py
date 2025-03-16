import streamlit as st
import os
import numpy as np

try:
    import faiss
    from langchain_community.vectorstores import FAISS
except ImportError:
    st.error("FAISS is not installed. Please install it using 'pip install faiss-cpu'.")
    faiss = None
    FAISS = None  # Avoid further errors

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain.llms import HuggingFaceHub
from langchain.retrievers import BM25Retriever
from transformers import pipeline

# Set the Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token is None:
    st.error("HUGGINGFACEHUB_API_TOKEN is not set. Please set it in your environment variables.")
    st.stop()

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
st.set_page_config(page_title="NLP Document Search", layout="wide")
st.title("NLP Document Search Application")
st.sidebar.title("Settings")
st.sidebar.header("Upload your PDF Documents")

# File uploader
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

pdf_files = []
if uploaded_files:
    pdf_files = [uploaded_file.name for uploaded_file in uploaded_files]
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

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

# Initialize FAISS index if FAISS is available
if faiss:
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        st.sidebar.success("FAISS index loaded from disk.")
    else:
        st.sidebar.warning("FAISS index not found. Rebuilding...")

        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)
        faiss.write_index(index, INDEX_PATH)
        st.sidebar.success("FAISS index saved to disk.")

    # Create FAISS vector store
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
    st.error("FAISS is not installed. Please install it using 'pip install faiss-cpu'.")
    retriever = None

# Define retrieval function
def ask_chatbot(question):
    if not retriever:
        return "Retrieval is not available because FAISS is missing."

    retrieved_docs = retriever.get_relevant_documents(question)

    if not retrieved_docs:
        return "No relevant information found."

    # Format Documents into Context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Generate Answer
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")
    response = llm_pipeline(f"Answer based on these documents:\n{context}\n\nQuestion: {question}", max_length=512, do_sample=True)
    
    return response[0]['generated_text']

# Main UI for question answering
st.header("Ask a Question")
question = st.text_input("Enter your question:")
if question:
    answer = ask_chatbot(question)
    st.write("### Answer")
    st.write(answer)

# Footer
st.markdown("---")
st.markdown("Created by [st125496](https://github.com/Laiba45362)")
