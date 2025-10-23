import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",  
)

documents = [
    "Python is a popular programming language used for data science, machine learning, and web development.",
    "Machine learning is a subset of artificial intelligence that focuses on training algorithms to make predictions.",
    "LangChain is a powerful framework for building LLM-based applications with memory, tools, and agents.",
    "Hugging Face provides state-of-the-art machine learning models and APIs for natural language processing.",
    "Streamlit is an open-source framework that helps you create interactive data apps using Python."
]

query = "Explain what Hugging Face does"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Calculate similarity scores both values should be 2D list
scores = cosine_similarity([query_embedding], doc_embeddings)[0]
print(cosine_similarity([query_embedding], doc_embeddings))

# Find the most similar document
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print("Query:", query)
print("Best Print resultsatch:", documents[index])
print("Similarity Score:", score)
