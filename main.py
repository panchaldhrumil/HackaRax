import os
import io
import json
import requests
import fitz  # PyMuPDF
from docx import Document
import email as email_parser
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import time

# Load environment variables from .env file
load_dotenv()

# --- Environment Variables ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN") 

PINECONE_INDEX_NAME = "insurance-policy-index"
LLM_MODEL = "gemini-1.5-flash-latest"

# --- CORRECT HUGGING FACE API URL ---
EMBEDDING_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-MiniLM-L-6-v3"

# Validate environment variables
if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, GEMINI_API_KEY, HF_API_TOKEN]):
    raise ValueError("One or more environment variables are not set. Please check all keys.")

print("All environment variables loaded successfully.")

# Initialize FastAPI app
app = FastAPI()

# --- Initialize Services ---
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

index = pc.Index(PINECONE_INDEX_NAME)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(LLM_MODEL)

# --- Data Models ---
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

# --- Helper Functions ---
def get_embeddings_from_api(texts: List[str]) -> List[List[float]]:
    """Gets embeddings by calling the Hugging Face Inference API."""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    response = requests.post(EMBEDDING_API_URL, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
    if response.status_code != 200:
        raise Exception(f"Hugging Face API request failed with status {response.status_code}: {response.text}")
    return response.json()

def parse_document(file_url: str):
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        content = response.content
        file_extension = file_url.split('?')[0].split('.')[-1].lower()
        text_chunks = []

        if file_extension == 'pdf':
            with fitz.open(stream=content, filetype='pdf') as doc:
                for page in doc:
                    chunks = [p.strip() for p in page.get_text("text").split('\n\n') if p.strip()]
                    text_chunks.extend(chunks)
        elif file_extension == 'docx':
            doc = Document(io.BytesIO(content))
            for para in doc.paragraphs:
                if para.text.strip():
                    text_chunks.append(para.text.strip())
        elif file_extension in ['msg', 'eml']:
            msg = email_parser.message_from_bytes(content)
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    payload = part.get_payload(decode=True)
                    if payload:
                        text_chunks.append(payload.decode('utf-8', errors='ignore').strip())
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        return [chunk for chunk in text_chunks if chunk]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing document: {str(e)}")

def get_llm_response(query: str, context: List[Dict]):
    context_str = "\n\n".join([f"Clause {i+1}: {c['text']}" for i, c in enumerate(context)])
    full_prompt = (
        "You are an expert insurance, legal, or HR analyst. Your task is to analyze the provided policy clauses "
        "and answer the user's question. Based *only* on the provided context, determine the answer. "
        "If the information is not present, state that clearly. Your response must be accurate, concise, "
        "and directly address the user's query."
        f"\n\nContext:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    try:
        response = gemini_model.generate_content(full_prompt)
        # Handle different response formats safely
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return str(response)
    except Exception as e:
        return f"An error occurred while processing the query: {str(e)}"

# --- API Endpoint ---
@app.post("/hackrx/run")
async def run_query_retrieval(request: RunRequest, authorization: str = Header(..., alias="Authorization")):
    expected_token = "Bearer 35928de76852eb7aacd2ad7b581bee5c8ab7539bdb514be752b6479293dccb2b"
    if authorization != expected_token:
        raise HTTPException(status_code=401, detail="Invalid or missing Authorization token.")

    try:
        text_chunks = parse_.document(request.documents)
        if not text_chunks:
            raise HTTPException(status_code=404, detail="Document content is empty or could not be parsed.")
        
        embeddings = get_embeddings_from_api(text_chunks)
        vectors = [(str(i), emb, {"text": text_chunks[i]}) for i, emb in enumerate(embeddings)]
        index.upsert(vectors=vectors, namespace=request.documents)

        answers = []
        for question in request.questions:
            query_embedding = get_embeddings_from_api([question])[0]
            search_results = index.query(vector=query_embedding, top_k=5, namespace=request.documents, include_metadata=True)
            relevant_clauses = [match['metadata'] for match in search_results['matches']]
            llm_answer = get_llm_response(question, relevant_clauses)
            answers.append(llm_answer)
        return {"answers": answers}
    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))