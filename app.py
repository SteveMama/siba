import streamlit as st
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# Available Open-source LLMs for local use
LLM_OPTIONS = {
    "GPT-Neo-2.7B": "EleutherAI/gpt-neo-2.7b",
    "MPT-7B-Instruct": "mosaicml/mpt-7b-instruct",
}

# Embedding model
EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"

# Streamlit page setup
st.set_page_config(
    page_title='Q&A Bot for PDF',
    page_icon='🔖',
    layout='wide'
)

if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = None
if "llm_pipeline" not in st.session_state:
    st.session_state["llm_pipeline"] = None


@st.cache_resource
def load_embedding(embedding_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(model_name=embedding_name, model_kwargs={"device": device})


@st.cache_resource
def load_llm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.7
    )
    return hf_pipeline


def process_pdf_to_vectorstore(pdf_path, embedding_model):
    loader = PDFPlumberLoader(str(pdf_path))
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    persist_directory = "chroma_db"
    os.makedirs(persist_directory, exist_ok=True)
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding_model, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb


def retrieve_context(vectordb, query, k=4):
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])


# Sidebar for inputs
with st.sidebar:
    emb_model = EMB_SBERT_MPNET_BASE
    llm_model = st.selectbox("**Select LLM Model**", list(LLM_OPTIONS.keys()), index=0)
    pdf_file = st.file_uploader("**Upload PDF**", type="pdf")
    if st.button("Submit") and pdf_file:
        with st.spinner("Processing PDF..."):
            try:
                with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    shutil.copyfileobj(pdf_file, tmp_file)
                    pdf_path = Path(tmp_file.name)

                embedding = load_embedding(emb_model)
                vectordb = process_pdf_to_vectorstore(pdf_path, embedding)
                llm_pipeline = load_llm(LLM_OPTIONS[llm_model])

                st.session_state["vectordb"] = vectordb
                st.session_state["llm_pipeline"] = llm_pipeline
                st.sidebar.success("PDF processed successfully!")
            except Exception as e:
                st.sidebar.error(f"Error processing PDF: {e}")

# Main interface
if st.session_state.get("vectordb") and st.session_state.get("llm_pipeline"):
    question = st.text_input("Ask a question about the document:", "")
    if st.button("Get Answer") and question:
        with st.spinner("Retrieving and generating answer..."):
            try:
                vectordb = st.session_state["vectordb"]
                llm_pipeline = st.session_state["llm_pipeline"]

                # Retrieve context
                context = retrieve_context(vectordb, question)

                # Generate answer
                prompt = f"""
                You are an AI assistant tasked with answering questions based on the following context:
                {context}

                Question: {question}

                Answer:
                """
                response = llm_pipeline(prompt, max_new_tokens=300, do_sample=False)
                st.write("**Answer:**")
                st.write(response[0]["generated_text"])
            except Exception as e:
                st.error(f"Error while generating answer: {e}")
else:
    st.write("Please upload a PDF to start.")
