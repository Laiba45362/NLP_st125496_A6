import streamlit as st
import os
import json
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFaceHub

# Set Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# Load the Hugging Face model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Function to load and process the PDF
@st.cache_resource
def load_vector_store():
    pdf_path = "Laiba_Muneer_resume.pdf"  # Ensure this file is in your GitHub repo
    if not os.path.exists(pdf_path):
        st.error("PDF file not found. Please upload 'Laiba_Muneer_resume.pdf' to the repository.")
        return None
    
    try:
        # Load the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_chunks = text_splitter.split_documents(documents)

        # Create embeddings using LangChain's SentenceTransformerEmbeddings
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create FAISS vector store directly from documents and embeddings
        vector_store = FAISS.from_documents(text_chunks, embedding_model)
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

# Initialize the vector store
vector_store = load_vector_store()

# Set up the QA chain
qa_chain = None
if vector_store:
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# Initialize session state to store responses
if "responses" not in st.session_state:
    st.session_state.responses = []

# Streamlit UI
st.title("Chatbot About Laiba Muneer")
st.write("Ask questions about Laiba Muneer based on their resume!")

# Input box for user question
question = st.text_input("Enter your question:")

# Button to submit question
if st.button("Ask"):
    if not question:
        st.write("Please enter a question.")
    elif not vector_store or not qa_chain:
        st.write("Cannot process questions due to missing PDF or initialization error.")
    else:
        try:
            # Get response from the QA chain
            response = qa_chain.invoke({"query": question})
            answer = response["result"]
            source_docs = response["source_documents"]

            # Display answer
            st.write("**Answer:**", answer)

            # Display source documents
            st.write("**Source Documents:**")
            for i, doc in enumerate(source_docs, 1):
                st.write(f"**Document {i}:** {doc.page_content[:200]}...")

            # Store the question and answer in session state
            st.session_state.responses.append({
                "question": question,
                "answer": answer
            })

        except Exception as e:
            st.write(f"Error processing question: {str(e)}")

# Display current responses
if st.session_state.responses:
    st.write("### Stored Responses")
    st.json(st.session_state.responses)

# End button to download JSON responses
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

