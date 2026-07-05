import chromadb
from data_loader import get_data
from chunker import get_chunks
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup Data
print("Loading and chunking data...")
raw_data = get_data()
chunks = get_chunks(raw_data)

# 2. Setup Embeddings
print("Loading embedding model...")
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Setup ChromaDB
print("Storing vectors in ChromaDB...")
client = chromadb.PersistentClient(path="./my_vector_db")
collection = client.create_collection(name="rag_docs", get_or_create=True)

# 4. Add to Database
# We create unique IDs for each chunk
ids = [str(i) for i in range(len(chunks))]
embeddings = model.embed_documents(chunks)

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=ids
)
print("Success! Data stored in ChromaDB.")

# 5. Search Test
query = "What is the Catholic character of the school?"
query_embedding = model.embed_query(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print(f"\nQuery: {query}")
print(f"Top result: {results['documents'][0][0]}")