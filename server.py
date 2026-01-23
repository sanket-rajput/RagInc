from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ Loading RAG Model...")
if os.path.exists("faiss_index_local"):
    # 1. Memory
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index_local", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    
    # 2. Brain (Local 4-bit Quantized Model)
    print("🔌 Loading Phi-3 in 4-bit mode (GPU Optimized)...")
    
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    
    # Configuration for 4-bit compression
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # Create Pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        return_full_text=False,
        temperature=0.3,
        do_sample=True, # Required for temperature
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # 3. Instructions
    system_prompt = (
        "You are the official assistant for PICT InC '26. "
        "Use the context below to answer questions accurately. "
        "If you don't know the answer, say 'I cannot find that info in the rulebook'. "
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("✅ Model Loaded on GPU!")
else:
    raise RuntimeError("❌ FAISS index not found. Run ingest.py first!")

class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        response = rag_chain.invoke({"input": request.question})
        return {"answer": response['answer']}
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))