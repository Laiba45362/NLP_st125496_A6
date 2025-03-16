
# AI Chatbot with FAISS and LangChain
**Deployment Link:** [Your Streamlit App](https://nlpst125496a6-nrf7qrhoh52ythv34a2c6x.streamlit.app/) 🚀
## Overview
This project demonstrates how to build an AI chatbot that uses FAISS (Facebook AI Similarity Search) for document indexing and LangChain for question-answering based on personal documents like PDFs. It integrates multiple language models from Hugging Face (Flan-T5 and GPT-2) and a potential Groq Cloud Llama LLM, with Streamlit for creating a user interface.

## Installation

Before running the code, ensure you have the required packages installed:

```bash
!pip install pypdf
!pip install -U langchain-community
!pip install faiss-cpu
!pip install sentence-transformers
!pip install streamlit
```

## Environment Setup

Set the Hugging Face API Token in your environment:

```python
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "---------------"
```

## Code Workflow

1. **Document Processing**:
   - PDF files are loaded using `PyPDFLoader`.
   - The text is split into chunks using `CharacterTextSplitter`.
   - The text content is then converted into embeddings using `SentenceTransformer`.

2. **FAISS Indexing**:
   - The embeddings are stored in a FAISS index, which is either loaded from a file or rebuilt if not found.
   - The FAISS index allows fast similarity-based search for retrieving relevant documents.

3. **Model Initialization**:
   - Hugging Face models (e.g., `google/flan-t5-large` and `gpt2`) are loaded using LangChain's `HuggingFaceHub`.
   - A potential third LLM (Groq Cloud Llama) can be used by replacing the placeholder.

4. **Question Answering**:
   - The user can ask questions via Streamlit.
   - The chatbot uses the LangChain `RetrievalQA` pipeline to answer the questions based on the relevant documents retrieved using FAISS.

5. **Streamlit Interface**:
   - The Streamlit app allows users to input a question and select a model.
   - The answer, along with the source documents, is displayed to the user.

## How to Run

1. Set the Hugging Face API token.
2. Load personal PDF documents.
3. Run the script to build the FAISS index (or load an existing one).
4. Run the Streamlit app to interact with the AI chatbot.

```bash
streamlit run app.py
```

## Code Explanation

- **FAISS Index**:
   - FAISS is used for fast similarity search by indexing document embeddings.
- **LangChain**:
   - LangChain integrates with various models and allows flexible chaining of models and retrievers.
- **Streamlit**:
   - Streamlit provides a simple interface to interact with the chatbot and get answers.

## Models

- **Hugging Face - google/flan-t5-large**: A large model suitable for question-answering tasks.
- **Hugging Face - GPT-2**: An alternative GPT model for generating responses.
- **Groq Cloud - Llama**: Placeholder for using the Groq Llama model (requires API key and endpoint).

## Potential Issues and Solutions

- **Irrelevant Document Retrieval**: Fine-tune the retriever model or improve document preprocessing to ensure better relevance.
- **Model Responses**: Ensure that the prompt design is clear and contextually informative to avoid generating irrelevant responses.

## Reference Documents
- Example PDF files are used to demonstrate the chatbot's functionality.

## License
This project is open-source. Feel free to contribute and improve the code!

