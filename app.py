import os
import boto3
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import pinecone
from rank_bm25 import BM25Okapi
from fastapi import FastAPI
from pydantic import BaseModel
import openai

# Initialize models and services
model = SentenceTransformer('all-MiniLM-L6-v2')
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI app
app = FastAPI()

# Data preparation
def download_from_s3(bucket_name, file_name):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, file_name, 'local_document.pdf')

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Embeddings and Pinecone
def generate_embeddings(chunks):
    return model.encode(chunks)

def upload_to_pinecone(index_name, chunks, embeddings):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=384)
    index = pinecone.Index(index_name)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        index.upsert([(f"chunk-{i}", embedding.tolist(), {"text": chunk})])

# Hybrid retrieval
def bm25_retrieval(query, tokenized_corpus, bm25, top_k=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [document_chunks[i] for i in top_indices]

def dense_retrieval(query, index, top_k=5):
    query_embedding = model.encode(query).tolist()
    results = index.query(query_embedding, top_k=top_k, include_metadata=True)
    return [result["metadata"]["text"] for result in results["matches"]]

def hybrid_retrieval(query, tokenized_corpus, bm25, index, top_k=5):
    bm25_results = bm25_retrieval(query, tokenized_corpus, bm25, top_k)
    dense_results = dense_retrieval(query, index, top_k)
    combined_results = list(set(bm25_results + dense_results))
    return combined_results[:top_k]

# OpenAI response generation
def generate_response(query, context):
    prompt = f"Query: {query}\nContext: {context}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# FastAPI endpoint
class Query(BaseModel):
    query: str

@app.post("/search")
def search(query: Query):
    # Load document chunks and BM25 index
    document_chunks = chunk_text(extract_text_from_pdf('local_document.pdf'))
    tokenized_corpus = [chunk.split() for chunk in document_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Generate embeddings and upload to Pinecone
    embeddings = generate_embeddings(document_chunks)
    upload_to_pinecone("enterprise-search", document_chunks, embeddings)

    # Perform hybrid retrieval
    index = pinecone.Index("enterprise-search")
    results = hybrid_retrieval(query.query, tokenized_corpus, bm25, index)

    # Generate AI response
    context = "\n".join(results)
    answer = generate_response(query.query, context)
    return {"results": results, "answer": answer}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
