import fitz  # PyMuPDF
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# Set Streamlit page configuration
st.set_page_config(page_title="Chat with Applicant's Docs", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome! Please upload any document of the Loan Applicant to get started."}]

if "initial" not in st.session_state:
    st.session_state.initial = False


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def stream_data(response_str):
    for word in response_str.split(" "):
        yield word + " "
        time.sleep(0.1)

def chunk_text(text, chunk_size=800, overlap=55):
    """Split text into smaller chunks for better embeddings."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# Title
st.title("Chat with Applicant's Docs")

# Display chat history
for msg in st.session_state.messages:
    if not msg["content"].startswith("Act as an experienced"):
        st.chat_message(msg["role"]).write(msg["content"])

# File upload
uploaded_file = st.sidebar.file_uploader("Upload a Document", type=["pdf"])

# File processing
if uploaded_file:
    with st.spinner("Uploading PDF & Processing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        # Extract text and process
        text = extract_text_from_pdf(temp_pdf_path)
        chunks = chunk_text(text)
        nodes = [TextNode(text=chunk) for chunk in chunks]
        index = VectorStoreIndex(nodes, show_progress=True)

        retriever = VectorIndexRetriever(index=index, similarity_top_k=12)
        response_synthesizer = get_response_synthesizer()

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)],
        )

        # Save to session
        st.session_state.text = text
        st.session_state.chunks = chunks
        st.session_state.nodes = nodes
        st.session_state.index = index
        st.session_state.retriever = retriever
        st.session_state.response_synthesizer = response_synthesizer
        st.session_state.query_engine = query_engine

    st.sidebar.success("PDF Uploaded & Processed Successfully!")

# Chat input and response
if prompt := st.chat_input(disabled=not uploaded_file):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Generating Response..."):
            response = st.session_state.query_engine.query(prompt)
            response_text = response.response if hasattr(response, "response") else str(response)

        st.write(stream_data(response_text))
        st.session_state.messages.append({"role": "assistant", "content": response_text})
