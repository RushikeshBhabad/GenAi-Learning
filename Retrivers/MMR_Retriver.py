"""
MMR (Maximal Marginal Relevance) Retriever Example
MMR helps retrieve diverse results, not just the most similar ones.
This prevents getting multiple nearly-identical documents.
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# Step 1: Create sample documents about web development
docs = [
    Document(page_content="React is a JavaScript library for building user interfaces."),
    Document(page_content="React helps developers create interactive web applications easily."),
    Document(page_content="Django is a Python web framework for rapid development."),
    Document(page_content="HTML and CSS are fundamental for web page structure and styling."),
    Document(page_content="Database systems store and manage application data efficiently."),
    Document(page_content="React, Angular, and Vue are popular frontend frameworks today."),
]

# Step 2: Initialize HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 3: Create FAISS vector store
print("Creating FAISS vector store...")
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# Step 4: Create MMR retriever
# search_type="mmr" enables Maximal Marginal Relevance
# k=3: retrieve top 3 results
# lambda_mult=0.5: balance between relevance (1.0) and diversity (0.0)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

# Step 5: Query the retriever
query = "What is React?"
print(f"\nQuery: {query}\n")

results = retriever.invoke(query)

# Display results
print(f"Retrieved {len(results)} diverse documents:\n")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(doc.page_content)
    print()

# Explanation
print("="*80)
print("Why MMR is useful:")
print("="*80)
print("""
Without MMR, you might get:
  1. React is a JavaScript library...
  2. React helps developers create...
  3. React, Angular, and Vue are...
  
With MMR, you get DIVERSE results:
  1. React is a JavaScript library... (most relevant)
  2. HTML and CSS are fundamental... (different but relevant)
  3. React, Angular, and Vue are... (diverse perspective)
  
MMR prevents redundant, nearly-identical results!
""")