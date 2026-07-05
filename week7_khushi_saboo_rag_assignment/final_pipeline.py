from data_loader import get_data
from chunker import get_chunks
from langchain_huggingface import HuggingFaceEmbeddings

print("Step 1: Fetching data...")
raw_data = get_data()

print("Step 2: Chunking...")
chunks = get_chunks(raw_data)

print(f"Step 3: Initializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print(f"Step 4: Generating embeddings for {len(chunks)} chunks (this might take a moment)...")
vectors = embeddings_model.embed_documents(chunks)

print(f"\nSUCCESS! Generated {len(vectors)} vectors.")
print(f"Sample vector (first 5 dimensions): {vectors[0][:5]}")