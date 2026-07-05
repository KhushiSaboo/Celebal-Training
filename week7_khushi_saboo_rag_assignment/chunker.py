from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_chunks(dataset):
    # Split text into 500 character chunks with a 50 character overlap
    # Overlap helps to maintain context between chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    all_chunks = []
    for row in dataset:
        # We process the 'context' column from your loaded data
        chunks = text_splitter.split_text(row['context'])
        all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":
    from data_loader import get_data
    raw_data = get_data()
    
    chunks = get_chunks(raw_data)
    print(f"\nTotal chunks generated: {len(chunks)}")
    print(f"Sample chunk: {chunks[0]}")