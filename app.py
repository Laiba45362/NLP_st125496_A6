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

# Set the Hugging Face API Token as an environment variable
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

# Initialize another Hugging Face LLM (e.g., GPT-2)
hf_llm_alternate = HuggingFaceHub(
    repo_id="gpt2",
    huggingfacehub_api_token=hf_token,
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Define FAISS index file path
INDEX_PATH = "faiss_index.bin"

# Initialize text_chunks to an empty list
text_chunks = []

# Initialize embedding_model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI setup
st.set_page_config(page_title="NLP Document Search", layout="wide")
st.title("NLP Document Search Application")
st.sidebar.title("Settings")
st.sidebar.header("Upload your PDF Documents")

# File uploader
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files is not None:
    pdf_files = [uploaded_file.name for uploaded_file in uploaded_files]
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
else:
    pdf_files = []

# Check if FAISS index exists and load it if available
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    st.sidebar.success("FAISS index loaded from disk.")
else:
    st.sidebar.warning("FAISS index not found. Rebuilding...")

    documents = []
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        else:
            st.error(f"{pdf_file} not found. Ensure the file exists.")

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)

    # Extract text content from chunks
    texts = [doc.page_content for doc in text_chunks]

    # Convert text to embeddings using SentenceTransformer
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)

    # Convert embeddings to numpy array for FAISS
    embedding_matrix = np.array(embeddings).astype("float32")

    # Initialize FAISS index
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    # Save FAISS index to disk
    faiss.write_index(index, INDEX_PATH)
    st.sidebar.success("FAISS index saved to disk.")

# Create FAISS vector store
docstore = InMemoryStore()
index_to_docstore_id = {}

document_objects = []
for i, doc in enumerate(text_chunks):
    doc_object = Document(page_content=doc.page_content, metadata=doc.metadata)
    doc_object.metadata['id'] = str(i)  # Add an 'id' to the metadata
    document_objects.append(doc_object)
    index_to_docstore_id[str(i)] = str(i)  # Ensure keys are strings

docstore.mset([(str(i), doc) for i, doc in enumerate(document_objects)])

vector_store = FAISS(
    embedding_function=embedding_model.encode,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Override `docstore.search` with `mget()`
def docstore_get(doc_id):
    docs = docstore.mget([doc_id])
    return docs[0] if docs else None

vector_store.docstore.search = docstore_get

# Setup Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Log Document Objects to Verify Structure
st.sidebar.write("### Document Objects")
for doc in document_objects:
    st.sidebar.write(f"Content: {doc.page_content[:100]}...")  # Print first 100 characters
    st.sidebar.write(f"Metadata: {doc.metadata}")

# Setup BM25 Keyword-Based Retriever
bm25_retriever = BM25Retriever.from_documents(document_objects)

# Hybrid Retrieval Function
def hybrid_retrieve(question):
    faiss_docs = retriever.get_relevant_documents(question)  # Dense retrieval
    bm25_docs = bm25_retriever.get_relevant_documents(question)  # Sparse retrieval
    
    # Combine both results (prioritize unique documents)
    combined_docs = {doc.page_content: doc for doc in faiss_docs + bm25_docs}.values()
    
    return list(combined_docs)

# Define query expansion for improved search results
def expand_query(question):
    alternative_queries = [
        f"What details are available about {question}?",
        f"Can you summarize information related to {question}?",
        f"Give me facts about {question}.",
        f"Explain {question} in simple terms.",
        f"What is known about {question} in the documents?"
    ]
    return [question] + alternative_queries

# Set up LangChain RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=hf_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Load FLAN-T5 Model Locally
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

# Function to Generate Answer Locally
def generate_answer_locally(question):
    response = llm_pipeline(question, max_length=512, do_sample=True)
    return response[0]['generated_text']

# Function to ask chatbot questions
def ask_chatbot(question):
    retrieved_docs = retriever.get_relevant_documents(question)
    
    if not retrieved_docs:
        return "No relevant information found.", []

    # Format Documents into Context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Ask LLM
    prompt = f"Answer based on these documents:\n{context}\n\nQuestion: {question}"
    answer = generate_answer_locally(prompt)

    return answer, retrieved_docs

# Main UI for question answering
st.header("Ask a Question")
question = st.text_input("Enter your question:")
if question:
    answer, retrieved_docs = ask_chatbot(question)
    st.write("### Answer")
    st.write(answer)
    
    st.write("### Retrieved Documents")
    for doc in retrieved_docs:
        st.write(f"Content: {doc.page_content[:500]}...")  # Show first 500 characters
        st.write(f"Metadata: {doc.metadata}")

# Footer
st.markdown("---")
st.markdown("Created by [st125496](https://github.com/Laiba45362)")
