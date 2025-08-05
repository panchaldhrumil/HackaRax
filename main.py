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
import pinecone 
from pinecone import ServerlessSpec
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, Column, Integer, Text, ARRAY, TIMESTAMP, String
from sqlalchemy import Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# --- Environment Variables ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN") # New Hugging Face Token

PINECONE_INDEX_NAME = "insurance-policy-index"
LLM_MODEL = "gemini-1.5-flash-latest"
EMBEDDING_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"

# Load DB creds
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

DATABASE_URL = os.environ.get("DATABASE_URL") 

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- QueryResult Model ---
class QueryResult(Base):
    __tablename__ = "INPUTS"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    document_url = Column(Text, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    context_clauses = Column(ARRAY(Text), nullable=False)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Validate environment variables
if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, GEMINI_API_KEY, HF_API_TOKEN]):
    raise ValueError("One or more environment variables are not set. Please check all keys.")

print("All environment variables loaded successfully.")

# Initialize FastAPI app
app = FastAPI(root_path="/api/v1")

# --- Initialize Services ---
pc = pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

index = pc.Index(PINECONE_INDEX_NAME)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(LLM_MODEL)

# Initialize SentenceTransformer embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Data Models ---
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

# --- Helper Functions ---
def get_embeddings_from_api(texts: List[str]) -> List[List[float]]:
    """Gets embeddings by calling the Hugging Face Inference API."""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    # For sentence-transformers models, we need to send each text separately
    embeddings = []
    for text in texts:
        response = requests.post(EMBEDDING_API_URL, headers=headers, json={"inputs": text})
        if response.status_code != 200:
            raise Exception(f"Hugging Face API request failed with status {response.status_code}: {response.text}")
        embeddings.append(response.json())
    return embeddings

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

def embed_and_index_document(document_url: str):
    """
    Parses, chunks, embeds, and indexes a document into Pinecone.
    """
    global index
    text_chunks = parse_document(document_url)

    if not text_chunks:
        raise HTTPException(status_code=404, detail="Document content is empty or could not be parsed.")

    # Create embeddings and index them in batches
    embeddings = embedding_model.encode(text_chunks).tolist()
    vectors = [(str(i), emb, {"text": text_chunks[i]}) for i, emb in enumerate(embeddings)]

    # Upsert vectors with namespace = document_url for separation
    index.upsert(vectors=vectors, namespace=document_url)
    print(f"Document from {document_url} indexed successfully with {len(vectors)} chunks.")

def get_llm_response(query: str, context: List[Dict]):
    context_str = "\n\n".join([f"Clause {i+1}: {c['text']}" for i, c in enumerate(context)])
    full_prompt = (
        "You are an expert insurance, legal, or HR analyst. Your task is to analyze the provided policy clauses "
        "and answer the user's question. Based *only* on the provided context, determine the answer. "
        "If the information is not present, state that clearly. Your response must be accurate, concise, "
        "and directly address the user's query. Do not include clause numbers or references in your response."
        f"\n\nContext:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    try:
        response = gemini_model.generate_content(full_prompt)
        # Handle different response formats safely
        if hasattr(response, 'text'):
            return response.text.replace('\n', ' ').strip()
        elif hasattr(response, 'candidates') and response.candidates:
            return response.candidates[0].content.parts[0].text.replace('\n', ' ').strip()
        else:
            return str(response).replace('\n', ' ').strip()
    except Exception as e:
        print(f"Error calling LLM: {str(e)}")
        # Check for response object details if available
        if 'response' in locals():
            print(f"Full response object: {str(response)}")
        return f"An error occurred while processing the query: {str(e)}"

# --- API Endpoint ---
@app.post("/hackrx/run")
async def run_query_retrieval(
    request: RunRequest,
    authorization: str = Header(..., alias="Authorization")
):
    # --- Authentication ---
    expected_token = "Bearer 35928de76852eb7aacd2ad7b581bee5c8ab7539bdb514be752b6479293dccb2b"
    if authorization != expected_token:
        raise HTTPException(status_code=401, detail="Invalid or missing Authorization token.")

    # --- Step 1 & 2: Process & Index Document ---
    try:
        embed_and_index_document(request.documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # --- Step 3-6: Process Queries ---
    answers = []
    db = SessionLocal()
    try:
        for question in request.questions:
            # Create embedding for the query
            query_embedding = embedding_model.encode(question).tolist()

            # Pinecone semantic search
            search_results = index.query(
                vector=query_embedding,
                top_k=5,
                namespace=request.documents,
                include_metadata=True
            )

            relevant_clauses = [match['metadata'] for match in search_results['matches']]

            # Generate answer from LLM
            llm_answer = get_llm_response(question, relevant_clauses)
            answers.append(llm_answer)

            # Store result in DB
            db_result = QueryResult(
                document_url=request.documents,
                question=question,
                answer=llm_answer,
                context_clauses=[c["text"] for c in relevant_clauses]
            )
            db.add(db_result)
            db.commit()
            db.refresh(db_result)

        print("Saved to DB")

    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        answers.append("An error occurred while processing one or more questions.")
    finally:
        db.close()

    # --- Step 7: Return JSON Output ---
    return {"answers": answers}