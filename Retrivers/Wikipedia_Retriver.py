"""
Wikipedia Retriever Example
This retriever fetches documents directly from Wikipedia based on your query.
"""

import os
from langchain_community.retrievers import WikipediaRetriever

# Set your HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "***REMOVED***"

# Initialize the Wikipedia retriever
# top_k_results: number of Wikipedia articles to retrieve
# lang: language of Wikipedia (en = English)
retriever = WikipediaRetriever(top_k_results=2, lang="en")

# Define your search query
query = "artificial intelligence and machine learning applications"

# Retrieve relevant Wikipedia documents
docs = retriever.invoke(query)

# Display the retrieved documents
print(f"Found {len(docs)} Wikipedia articles:\n")
for i, doc in enumerate(docs):
    print(f"\n{'='*80}")
    print(f"Article {i+1}: {doc.metadata.get('title', 'Unknown')}")
    print(f"{'='*80}")
    # Display first 500 characters of content
    print(f"{doc.page_content[:500]}...")
    print(f"\nSource: {doc.metadata.get('source', 'N/A')}")