import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM 

# 1. Setup
client = chromadb.PersistentClient(path="./my_vector_db")
collection = client.get_collection(name="rag_docs")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = OllamaLLM(model="llama3") # Local LLM loaded

def get_answer(question):
    # Retrieval
    query_embedding = embedding_model.embed_query(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=2)
    context = "\n".join(results['documents'][0])
    
    # Generation Prompt
    prompt = f"""
    Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    Answer:
    """
    
    # LLM ko prompt bhejna
    response = llm.invoke(prompt)
    return response

if __name__ == "__main__":
    q = input("Ask a question: ")
    print("Thinking...")
    print("\nAI Answer:", get_answer(q))