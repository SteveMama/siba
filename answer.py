from extract import *
from embed import *
from groq import Groq


def answer_question(question, context, groq_api_key):
    client = Groq(api_key="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")

    system_prompt = f"""Answer the question based on this context: {context}
    If you cannot answer from the context, say 'I don't have enough information.'"""

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.1,
        max_tokens=1024
    )

    return completion.choices[0].message.content


def query_document(question, index, chunks, model, groq_api_key, k=3):
    # Create question embedding
    question_embedding = model.encode([question])

    # Search similar chunks
    D, I = index.search(question_embedding.astype('float32'), k)

    # Get relevant context
    context = " ".join([chunks[i] for i in I[0]])

    # Get answer using Groq
    return answer_question(question, context, groq_api_key)


def main(pdf_path, groq_api_key):
    # Extract text
    text = extract_text_from_pdf(pdf_path)

    # Create index and chunks
    index, chunks = create_embeddings(text)

    # Initialize sentence transformer
    model = SentenceTransformer('all-mpnet-base-v2')

    # Interactive question answering
    while True:
        question = input("Ask a question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        answer = query_document(question, index, chunks, model, groq_api_key)
        print(f"Answer: {answer}\n")

