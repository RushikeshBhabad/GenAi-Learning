"""
Vector Store Retriever Example
This retriever stores documents as embeddings and searches for similar content.
Uses HuggingFace embeddings instead of OpenAI.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Set your HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "***REMOVED***"

# Step 1: Create sample documents about programming
documents = [
    Document(page_content="Python is a high-level programming language known for simplicity."),
    Document(page_content="Vector databases store data as numerical embeddings for fast search."),
    Document(page_content="Machine learning models convert text into numerical vectors."),
    Document(page_content="HuggingFace provides pre-trained models for various NLP tasks."),
]

# Step 2: Initialize HuggingFace embedding model
# Using 'sentence-transformers/all-MiniLM-L6-v2' - a lightweight, fast model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 3: Create Chroma vector store in memory
print("Creating vector store...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="programming_docs"
)

# Step 4: Convert vectorstore into a retriever
# k=2 means retrieve top 2 most similar documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Step 5: Query the retriever
query = "What is a vector database?"
print(f"\nQuery: {query}\n")

results = retriever.invoke(query)

# Display results
print(f"Retrieved {len(results)} relevant documents:\n")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(doc.page_content)
    print()

# Alternative: Direct similarity search (without creating retriever)
print("\n" + "="*80)
print("Alternative: Direct similarity search")
print("="*80)
results_direct = vectorstore.similarity_search(query, k=2)

for i, doc in enumerate(results_direct):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)