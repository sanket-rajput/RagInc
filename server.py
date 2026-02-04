"""
PICT Event RAG - Smart Short Answers
Uses Groq API (free) for better responses
"""

import os
import re
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import httpx
import uvicorn

warnings.filterwarnings("ignore")

# Config
DATA_DIR = "./pict_event_data"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Free Groq API - get key from https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"

# Global
vectorstore = None
info = {"docs": 0, "chunks": 0, "status": "loading"}

def extract_answer(question: str, context: str) -> str:
    """Extract concise answer from context based on question keywords."""
    
    q_lower = question.lower()
    lines = [l.strip() for l in context.split('\n') if l.strip()]
    
    # Common question patterns
    if any(w in q_lower for w in ['date', 'when', 'time']):
        dates = []
        for line in lines:
            if any(w in line for w in ['March', 'march', '2026', '20 ', '21 ', '22 ']):
                dates.append(line)
        if dates:
            return "PICT InC 2026 will be held on 20, 21, and 22 March 2026."
    
    if any(w in q_lower for w in ['fee', 'cost', 'price', 'registration']):
        if 'concepts' in q_lower:
            return "CONCEPTS registration fee is Rs. 500 per team."
        if 'impetus' in q_lower:
            return "IMPETUS registration fee is Rs. 100 per team."
        return "Registration fees: CONCEPTS (Final Year) - Rs. 500/team, IMPETUS (FE/SE/TE) - Rs. 100/team."
    
    if any(w in q_lower for w in ['eligibility', 'who can', 'eligible']):
        for line in lines:
            if 'eligib' in line.lower() or 'final year' in line.lower() or 'fe, se' in line.lower():
                return line
    
    if any(w in q_lower for w in ['venue', 'where', 'location', 'address']):
        return "Venue: Pune Institute of Computer Technology (PICT), Sr. No. 27, Near Trimurti Chowk, Dhankawadi, Pune 411043."
    
    if any(w in q_lower for w in ['team', 'members', 'size']):
        for line in lines:
            if 'team' in line.lower() or 'member' in line.lower() or 'minimum' in line.lower() or 'maximum' in line.lower():
                return line
    
    if any(w in q_lower for w in ['what is', 'about', 'explain']):
        for line in lines:
            if 'inc' in line.lower() or 'impetus' in line.lower() or 'concepts' in line.lower():
                if len(line) > 30:
                    return line[:250] + "..." if len(line) > 250 else line
    
    if any(w in q_lower for w in ['event', 'events']):
        return "InC 2026 has 4 main events: CONCEPTS (Final Year), IMPETUS (FE/SE/TE), PRADNYA (Programming), and TECHFIESTA (Hackathon)."
    
    if any(w in q_lower for w in ['domain', 'topics', 'categories']):
        for line in lines:
            if 'domain' in line.lower() or 'application' in line.lower() or 'machine learning' in line.lower():
                return line
    
    if 'concepts' in q_lower:
        return "CONCEPTS is the Final Year (BE) project exhibition. Fee: Rs. 500/team. Team size: 1-5 members."
    
    if 'impetus' in q_lower:
        return "IMPETUS is for FE, SE, TE students. Fee: Rs. 100/team. Team size: 1-5 members."
    
    if 'pradnya' in q_lower:
        return "PRADNYA is a programming contest testing programming fundamentals, logic, and problem-solving."
    
    # Default: return best matching content
    for line in lines:
        if len(line) > 30 and not line.startswith('Alright'):
            return line[:200] + "..." if len(line) > 200 else line
    
    return "I found some info but couldn't extract a specific answer. Try asking more specifically."

async def ask_groq(question: str, context: str) -> str:
    """Use Groq API for smart answers."""
    if not GROQ_API_KEY:
        return extract_answer(question, context)
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are PICT Event Assistant. Answer in 1-2 short sentences using ONLY the context. If not found, say 'I don't have that info.'"},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.1
                },
                timeout=10
            )
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except:
        return extract_answer(question, context)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, info
    
    print("\n🚀 Starting PICT RAG...")
    
    # Load embeddings
    print("🧠 Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )
    
    # Load documents
    print(f"📄 Loading documents from {DATA_DIR}...")
    docs = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".txt"):
            with open(os.path.join(DATA_DIR, f), "r") as file:
                docs.append(Document(page_content=file.read(), metadata={"source": f}))
    
    info["docs"] = len(docs)
    
    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = splitter.split_documents(docs)
        info["chunks"] = len(chunks)
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        info["status"] = "ready"
        print(f"✅ Ready! {info['docs']} docs → {info['chunks']} chunks")
        print(f"🔑 Groq API: {'Enabled' if GROQ_API_KEY else 'Disabled (using local extraction)'}\n")
    else:
        info["status"] = "no_docs"
        print("⚠️ No documents found!\n")
    
    yield
    print("🛑 Shutting down...")

# App
app = FastAPI(title="PICT RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.get("/health")
async def health():
    return info

@app.post("/chat")
async def chat(req: ChatRequest):
    if not vectorstore:
        return {"answer": "System not ready yet."}
    
    try:
        # Search for relevant chunks
        results = vectorstore.similarity_search(req.question, k=2)
        
        if not results:
            return {"answer": "I don't have information about that."}
        
        # Combine context
        context = "\n".join([doc.page_content for doc in results])
        
        # Get smart answer
        answer = await ask_groq(req.question, context)
        
        return {"answer": answer}
        
    except Exception as e:
        return {"answer": f"Error: {str(e)[:100]}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
