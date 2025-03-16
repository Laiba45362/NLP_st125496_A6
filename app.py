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

# Set API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("Hugging Face API token not found! Set it in environment variables.")

# Load LLM
hf_llm = HuggingFaceHub(repo_id="google/flan-t5-large", huggingfacehub_api_token=hf_token, model_kwargs={"temperature": 0.7, "max_length": 512})

# Initialize FAISS index
INDEX_PATH = "faiss_index.bin"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_pdf_files(uploaded_files):
    documents = []
    for file in uploaded_files:
        loader = PyPDFLoader(file.name)
        documents.extend(loader.load())
    return documents

def create_faiss_index(text_chunks):
    texts = [doc.page_content for doc in text_chunks]
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    embedding_matrix = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    faiss.write_index(index, INDEX_PATH)
    return index

# Streamlit UI Design
st.set_page_config(page_title="Smart PDF Search", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Smart NLP Document Search</h1>", unsafe_allow_html=True)
st.sidebar.header("Upload PDF Documents")

uploaded_files = st.sidebar.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)

documents = load_pdf_files(uploaded_files) if uploaded_files else []
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)

docstore = InMemoryStore()
document_objects = [Document(page_content=doc.page_content, metadata={"id": str(i)}) for i, doc in enumerate(text_chunks)]
docstore.mset([(str(i), doc) for i, doc in enumerate(document_objects)])

if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = create_faiss_index(text_chunks)

vector_store = FAISS(embedding_function=embedding_model.encode, index=index, docstore=docstore, index_to_docstore_id={str(i): str(i) for i in range(len(document_objects))})
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def ask_chatbot(question):
    retrieved_docs = retriever.get_relevant_documents(question)
    if not retrieved_docs:
        return "No relevant information found.", []
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"Answer based on these documents:\n{context}\n\nQuestion: {question}"
    return hf_llm(prompt), retrieved_docs

st.markdown("### Ask a Question")
question = st.text_input("Enter your query:")
if question:
    answer, retrieved_docs = ask_chatbot(question)
    st.markdown("#### Answer")
    st.write(answer)
    st.markdown("#### Retrieved Documents")
    for doc in retrieved_docs:
        st.write(f"{doc.page_content[:500]}...")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Created by <a href='https://github.com/Laiba45362'>st125496</a></p>", unsafe_allow_html=True)
