# AI-Powered Enterprise Search System

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** search system using **hybrid retrieval (BM25 + dense embeddings)** and OpenAI’s **GPT model** to improve internal document search accuracy. The system processes PDF documents, indexes them in **Pinecone**, retrieves relevant content, and generates AI-powered responses.

## Features
**Hybrid Retrieval** (BM25 for keyword matching + Dense embeddings for semantic search)
 **Text Extraction from PDFs** using PyPDF2
 **Vector Storage & Search** with Pinecone
 **AI Response Generation** with OpenAI GPT-3
 **FastAPI Endpoint** for easy query processing

## Technologies Used
- **Programming Language:** Python
- **NLP & Embeddings:** Sentence Transformers (all-MiniLM-L6-v2), OpenAI API
- **Vector Database:** Pinecone
- **Keyword Search:** BM25 (rank_bm25)
- **Frameworks:** FastAPI
- **Cloud Services:** AWS S3 (for document storage)

## Installation

### 1. Clone the Repository
```sh
 git clone https://github.com/your-repo/ai-enterprise-search.git
 cd ai-enterprise-search
```

### 2. Install Dependencies
```sh
 pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file and add the following keys:
```sh
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

## Usage

### 1. Run the API
```sh
uvicorn main:app --reload
```

### 2. Send a Search Query
You can test the search system using a **POST request**:
```sh
curl -X POST "http://127.0.0.1:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the company policy on remote work?"}'
```

### 3. Expected Response
```json
{
  "results": ["Relevant text from document"],
  "answer": "AI-generated response based on retrieved text."
}
```

## How It Works
1. **Document Processing:** Downloads PDFs from AWS S3 and extracts text.
2. **Chunking:** Splits text into smaller chunks (500 characters).
3. **Embedding Generation:** Converts chunks into numerical vectors using **Sentence Transformers**.
4. **Indexing & Retrieval:**
   - **BM25:** Retrieves text chunks using keyword matching.
   - **Pinecone:** Retrieves embeddings using semantic search.
5. **Response Generation:** Passes retrieved text to **OpenAI GPT-3** for natural language answers.

## Future Enhancements
- 🔹 Integrate **multi-modal search** (text + images)
- 🔹 Fine-tune **LLMs** for better domain-specific responses
- 🔹 Implement **user authentication** for secure access

## Contributors
- **Ghulam Rabbani** – AI Engineer

