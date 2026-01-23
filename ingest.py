import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_data():
    print("🔄 Loading PDF files from 'data' folder...")
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages.")

    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    text_chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(text_chunks)} text chunks.")

    print("🧠 Creating Local Vector Database with HuggingFace...")
    # This runs locally on your machine
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    vector_store.save_local("faiss_index_local")
    print("🎉 Success! Local knowledge base saved to 'faiss_index_local'.")

if __name__ == "__main__":
    ingest_data()