from datasets import load_dataset

def get_data():
    print("Connecting to Hugging Face...")
    # correct repo path 
    dataset = load_dataset("rajpurkar/squad", split="train[:100]")
    return dataset

if __name__ == "__main__":
    try:
        data = get_data()
        print("\nDataset successfully loaded!")
        print(f"Columns: {data.column_names}")
        print(f"Sample Context: {data[0]['context'][:200]}...")
    except Exception as e:
        print(f"\nError: {e}")