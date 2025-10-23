"""
Multi-Query Retriever Example
This retriever generates multiple variations of your query to find better results.
Uses HuggingFace LLM instead of OpenAI.
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load environment variables from .env file
load_dotenv()

# Step 1: Create diverse documents about fitness and technology
all_docs = [
    Document(page_content="Daily exercise improves cardiovascular health and mental wellbeing.", metadata={"source": "F1"}),
    Document(page_content="Green vegetables and fresh fruits provide essential vitamins and antioxidants.", metadata={"source": "F2"}),
    Document(page_content="Quality sleep helps the body recover and strengthens immune function.", metadata={"source": "F3"}),
    Document(page_content="Meditation and yoga reduce stress and enhance focus and concentration.", metadata={"source": "F4"}),
    Document(page_content="Staying hydrated supports digestion and maintains body temperature.", metadata={"source": "F5"}),
    Document(page_content="Cloud computing enables scalable storage and processing of data.", metadata={"source": "T1"}),
    Document(page_content="JavaScript offers flexibility for both frontend and backend development.", metadata={"source": "T2"}),
    Document(page_content="Solar panels convert sunlight into renewable electrical power.", metadata={"source": "T3"}),
    Document(page_content="The Olympics bring together athletes from nations worldwide.", metadata={"source": "T4"}),
    Document(page_content="Quantum computers process information using quantum mechanics principles.", metadata={"source": "T5"}),
]

# Step 2: Initialize HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 3: Create FAISS vector store
print("Creating vector store...")
vectorstore = FAISS.from_documents(documents=all_docs, embedding=embedding_model)

# Step 4: Create standard similarity retriever
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5}
)

# Step 5: Create MultiQuery retriever with HuggingFace LLM
# Using a small, fast model for query generation
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    temperature=0.7,
    max_length=512
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)

# Step 6: Query both retrievers
query = "How to boost vitality and stay healthy?"

print(f"\nOriginal Query: {query}\n")
print("="*80)

# Retrieve with standard similarity search
print("\nSTANDARD SIMILARITY SEARCH RESULTS:")
print("-"*80)
similarity_results = similarity_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"\nResult {i+1} [{doc.metadata['source']}]:")
    print(doc.page_content)

# Retrieve with multi-query approach
print("\n" + "="*80)
print("MULTI-QUERY RETRIEVER RESULTS:")
print("-"*80)
print("(Generates multiple query variations for better coverage)\n")

multiquery_results = multiquery_retriever.invoke(query)

for i, doc in enumerate(multiquery_results):
    print(f"\nResult {i+1} [{doc.metadata['source']}]:")
    print(doc.page_content)

# Explanation
print("\n" + "="*80)
print("How Multi-Query Works:")
print("="*80)
print("""
1. Takes your original query
2. LLM generates multiple query variations:
   - "How to boost vitality and stay healthy?"
   - "Ways to increase energy levels naturally"
   - "Tips for maintaining good health"
   
3. Searches with ALL variations
4. Combines and de-duplicates results
5. Returns more comprehensive, diverse answers!
""")