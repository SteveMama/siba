import os
import PyPDF2
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
from yt_search import *
from groq import Groq

# Initialize the Groq API client
client = Groq(api_key="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")

# Load the embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
model = AutoModel.from_pretrained(embedding_model_name).to('cpu')

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="vectordb")
collection = chroma_client.get_collection("all_pdfs")


# Define helper functions
def encode_question(question):
    """Convert a question into an embedding."""
    inputs = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.cpu().numpy().tolist()[0]


def search_similar_chunks(question_embedding, top_k=10):
    """Search for the top k similar chunks in the ChromaDB collection."""
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    return list(zip(results['documents'][0], results['metadatas'][0], results['distances'][0]))


def search_youtube_videos_with_transcripts(query, top_n=3):
    """Search for YouTube videos and retrieve their transcripts."""
    query = " tony seba and " + query
    youtube_results = search_youtube(query, max_results=top_n)
    results = []
    for video in youtube_results:
        transcript = get_transcript(video['video_id'])
        relevance = calculate_relevance(query, video['title'] + " " + video['description'] + " " + transcript)
        results.append({
            'title': video['title'],
            'url': video['url'],
            'relevance_score': relevance,
            'transcript': transcript[:1000] + '...' if len(transcript) > 1000 else transcript  # Truncate long transcripts
        })
    # Sort by relevance
    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    return results

def answer_question(question):
    try:
        # Encode the question
        question_embedding = encode_question(question)

        # Search for similar chunks
        similar_chunks = search_similar_chunks(question_embedding)

        # Retrieve the content of the top chunks
        context = ""
        for chunk_content, metadata, distance in similar_chunks:
            context += f"From {metadata['source']}, chunk {metadata['chunk_id']}:\n{chunk_content}\n\n"

        print(context)
        # Search for YouTube videos and transcripts
        youtube_results = search_youtube_videos_with_transcripts(question, top_n=3)
        for video in youtube_results:
            context += f"From YouTube video '{video['title']}' ({video['url']}):\n{video['transcript']}\n\n"

        print(context)

        # Prepare the prompt for the LLM
        prompt = f"""Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the question: {question}
        """

        # Use Groq API to get the answer
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the given context. If you don't find relevant information, reply with 'I do not have information'.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=500,
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI
st.title("PDF and YouTube Question Answering with AI")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
user_input = st.text_input("Ask a question about the documents or YouTube content:", key="user_input")
if st.button("Send") and user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate the answer
    answer = answer_question(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display the chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").markdown(msg["content"])