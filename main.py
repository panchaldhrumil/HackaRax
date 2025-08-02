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
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file

print("🟢 Starting app...")

from fastapi import FastAPI, Header, HTTPException
from pinecone import pinecone, ServerlessSpec

print("🟢 Initializing Pinecone...")
pinecone.init(PINECONE_API_KEY="your_pinecone_key", PINECONE_ENV="your_pinecone_environment")
print("✅ Pinecone initialized")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "App is running"}

load_dotenv()

# --- Environment Variables ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_INDEX_NAME = "insurance-policy-index"
LLM_MODEL = "gemini-1.5-flash-latest"

# Validate environment variables
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT environment variable is not set")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

print(f"Loaded PINECONE_API_KEY: {PINECONE_API_KEY[:4]}...{PINECONE_API_KEY[-4:]}")
print(f"Loaded PINECONE_ENVIRONMENT: {PINECONE_ENVIRONMENT}")
print(f"Loaded GEMINI_API_KEY: {GEMINI_API_KEY[:4]}...{GEMINI_API_KEY[-4:]}")




# --- Initialize Services ---

# Initialize Pinecone client instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # all-MiniLM-L6-v2 embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Initialize Gemini LLM
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(LLM_MODEL)

# Initialize SentenceTransformer embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Data Models ---
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

# --- Helper Functions ---
def parse_document(file_url: str):
    """
    Parses documents from a URL (PDF, DOCX, etc.) and extracts text chunks.
    Returns a list of text chunks.
    """
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        content = response.content

        file_extension = file_url.split('?')[0].split('.')[-1].lower()
        text_chunks = []

        if file_extension == 'pdf':
            with fitz.open(stream=content, filetype='pdf') as doc:
                for page in doc:
                    page_text = page.get_text("text")
                    chunks = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                    text_chunks.extend(chunks)
        elif file_extension == 'docx':
            doc = Document(io.BytesIO(content))
            for para in doc.paragraphs:
                if para.text.strip():
                    text_chunks.append(para.text.strip())
        elif file_extension in ['msg', 'eml']:
            msg = email_parser.message_from_bytes(content)
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    if ctype == 'text/plain':
                        payload = part.get_payload(decode=True)
                        if payload:
                            text_chunks.append(payload.decode('utf-8', errors='ignore').strip())
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    text_chunks.append(payload.decode('utf-8', errors='ignore').strip())
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return [chunk for chunk in text_chunks if chunk]

    except Exception as e:
        print(f"Error parsing document: {e}")
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
    """
    Uses Gemini LLM to generate an answer based on the query and retrieved context.
    """
    context_str = "\n\n".join([f"Clause {i+1}: {c['text']}" for i, c in enumerate(context)])

    full_prompt = (
        "You are an expert insurance, legal, or HR analyst. Your task is to analyze the provided policy clauses "
        "and answer the user's question. Based *only* on the provided context, determine the answer. "
        "If the information is not present, state that clearly. Your response must be accurate, concise, "
        "and directly address the user's query. After your answer, provide the number of the specific clause(s) "
        "that you used to form your response."
        f"\n\nContext:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    try:
        response = gemini_model.generate_content(full_prompt)
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return str(response)
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

    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        answers.append("An error occurred while processing one or more questions.")

    # --- Step 7: Return JSON Output ---
    return {"answers": answers}