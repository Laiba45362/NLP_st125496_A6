import streamlit as st
import os
import json
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFaceHub

# Set the Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load the Hugging Face model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Streamlit UI setup
st.set_page_config(page_title="📜 AI Document Search", layout="wide")
st.title("🔍 AI-Powered Document Search")
st.markdown("Find answers in your documents with AI!")

# Sidebar for file upload
st.sidebar.title("⚙️ Upload PDF Documents")
uploaded_files = st.sidebar.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)

# Load and process PDFs
@st.cache_resource
def load_vector_store(uploaded_files):
    if not uploaded_files:
        return None
    
    documents = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(uploaded_file.name)
        documents.extend(loader.load())
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(text_chunks, embedding_model)
    return vector_store

vector_store = load_vector_store(uploaded_files)

# Initialize RetrievalQA
qa_chain = None
if vector_store:
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# Ask a question
st.header("💡 Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Ask"):
    if not question:
        st.warning("Please enter a question.")
    elif not vector_store or not qa_chain:
        st.error("No document data available. Upload PDFs first.")
    else:
        try:
            response = qa_chain.invoke({"query": question})
            answer = response["result"]
            source_docs = response["source_documents"]
            st.success("✅ Answer:")
            st.write(answer)
            st.write("**Source Documents:**")
            for i, doc in enumerate(source_docs, 1):
                st.write(f"**Document {i}:** {doc.page_content[:200]}...")
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
