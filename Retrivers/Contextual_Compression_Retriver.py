"""
Contextual Compression Retriever Example
This retriever filters and compresses retrieved documents to include only relevant information.
Uses HuggingFace LLM to extract only the parts that answer the query.
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# Step 1: Create documents with mixed relevant/irrelevant content
docs = [
    Document(page_content="""
        The Eiffel Tower is a famous landmark in Paris, France.
        Renewable energy sources include solar panels and wind turbines.
        Millions of tourists visit Paris annually. The tower was built in 1889.
    """, metadata={"source": "Doc1"}),
    
    Document(page_content="""
        Ancient Rome was a powerful civilization in Mediterranean history.
        Wind turbines generate electricity by converting kinetic energy.
        Gladiators fought in the Colosseum. Roman engineering was advanced.
    """, metadata={"source": "Doc2"}),
    
    Document(page_content="""
        Football is the world's most popular sport with billions of fans.
        It originated in England in the 19th century. FIFA organizes World Cup.
    """, metadata={"source": "Doc3"}),
    
    Document(page_content="""
        Modern cinema began in the early 1900s with silent black-and-white films.
        Directors like Chaplin pioneered filmmaking. Solar panels convert sunlight to electricity.
        Contemporary movies use advanced CGI and surround sound technology.
    """, metadata={"source": "Doc4"})
]

# Step 2: Initialize HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 3: Create FAISS vector store
print("Creating vector store...")
vectorstore = FAISS.from_documents(docs, embedding_model)

# Step 4: Create base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Step 5: Set up HuggingFace LLM for compression
print("Initializing LLM compressor...")
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    temperature=0.1,
    max_length=512
)

compressor = LLMChainExtractor.from_llm(llm)

# Step 6: Create contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Step 7: Query the retriever
query = "What are renewable energy sources?"
print(f"\nQuery: {query}\n")
print("="*80)

# Get compressed results
print("\nCOMPRESSED RESULTS (only relevant parts):")
print("-"*80)
compressed_results = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_results):
    print(f"\nResult {i+1} [from {doc.metadata['source']}]:")
    print(doc.page_content.strip())

# Compare with uncompressed results
print("\n" + "="*80)
print("UNCOMPRESSED RESULTS (full documents):")
print("-"*80)
uncompressed_results = base_retriever.invoke(query)

for i, doc in enumerate(uncompressed_results):
    print(f"\nResult {i+1} [from {doc.metadata['source']}]:")
    print(doc.page_content.strip())

# Explanation
print("\n" + "="*80)
print("Benefits of Contextual Compression:")
print("="*80)
print("""
WITHOUT Compression:
  - Returns entire documents with lots of irrelevant content
  - Harder to find the actual answer
  - Wastes tokens in LLM context window
  
WITH Compression:
  - Extracts ONLY the relevant sentences
  - Clean, focused results
  - Saves tokens and improves accuracy
  
Perfect for RAG (Retrieval-Augmented Generation) systems!
""")