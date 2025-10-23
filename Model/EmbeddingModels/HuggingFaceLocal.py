from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Machine learning is a subset of artificial intelligence focused on building systems that learn from data.",
    "The Eiffel Tower in Paris attracts millions of tourists every year and is one of the most visited landmarks in the world.",
    "The human heart pumps blood throughout the body, supplying oxygen and nutrients to various organs."
]

# Generate embeddings for the documents
vector = embedding.embed_documents(documents)

print("\nGenerated Embeddings:\n")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {doc}")
    print(f"Embedding Vector (first 10 dims): {vector[i][:10]}\n")
