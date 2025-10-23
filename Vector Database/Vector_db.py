# Setup
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "***REMOVED***"

# Install required packages
# !pip install langchain chromadb sentence-transformers langchain-huggingface langchain-community

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create LangChain documents for travel destinations
doc1 = Document(
    page_content="Paris, the City of Light, is famous for the Eiffel Tower, Louvre Museum, and its romantic atmosphere. Known for world-class cuisine, fashion, and art, it offers charming cafes, historic architecture, and the beautiful Seine River for scenic walks.",
    metadata={"country": "France", "type": "city"}
)

doc2 = Document(
    page_content="Bali, Indonesia is a tropical paradise known for its stunning beaches, lush rice terraces, and ancient temples. Popular for surfing, yoga retreats, and vibrant culture, it offers a perfect blend of relaxation and adventure.",
    metadata={"country": "Indonesia", "type": "island"}
)

doc3 = Document(
    page_content="Tokyo, Japan seamlessly blends ultra-modern technology with traditional culture. From bustling Shibuya Crossing to serene temples, visitors can enjoy incredible food, cherry blossoms, and efficient transportation throughout this dynamic metropolis.",
    metadata={"country": "Japan", "type": "city"}
)

doc4 = Document(
    page_content="The Swiss Alps offer breathtaking mountain scenery perfect for skiing, hiking, and mountaineering. Known for pristine lakes, charming villages like Zermatt and Interlaken, Switzerland provides year-round outdoor adventures and stunning natural beauty.",
    metadata={"country": "Switzerland", "type": "mountain"}
)

doc5 = Document(
    page_content="Santorini, Greece features iconic white-washed buildings with blue domes overlooking the Aegean Sea. Famous for spectacular sunsets, volcanic beaches, and ancient ruins, this island destination offers Mediterranean charm and excellent local wine.",
    metadata={"country": "Greece", "type": "island"}
)

# Combine documents
docs = [doc1, doc2, doc3, doc4, doc5]

# Create vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='travel_chroma_db',
    collection_name='destinations'
)

# Add documents
print("Adding documents...")
doc_ids = vector_store.add_documents(docs)
print(f"Added documents with IDs: {doc_ids}")

# View all documents
print("\n=== All Documents ===")
all_docs = vector_store.get(include=['documents', 'metadatas'])
for i, (doc, meta) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
    print(f"\nDoc {i+1} ({meta}):\n{doc[:100]}...")

# Search documents
print("\n=== Similarity Search: Beach destinations ===")
results = vector_store.similarity_search(
    query='Where can I find beautiful beaches?',
    k=2
)
for doc in results:
    print(f"\n{doc.metadata}: {doc.page_content}")

# Search with similarity score
print("\n=== Similarity Search with Scores: Mountain activities ===")
results_with_scores = vector_store.similarity_search_with_score(
    query='Where can I go hiking in mountains?',
    k=2
)
for doc, score in results_with_scores:
    print(f"\nScore: {score:.4f}")
    print(f"{doc.metadata}: {doc.page_content}")

# Metadata filtering
print("\n=== Filter by Country: Greece ===")
filtered_results = vector_store.similarity_search_with_score(
    query="",
    filter={"country": "Greece"}
)
for doc, score in filtered_results:
    print(f"\nScore: {score:.4f}")
    print(f"{doc.metadata}: {doc.page_content}")

# Update a document
print("\n=== Updating Document ===")
updated_doc1 = Document(
    page_content="Paris, the capital of France, is one of the world's most visited cities. Beyond the iconic Eiffel Tower, visitors can explore the Louvre Museum housing the Mona Lisa, stroll through Montmartre's artistic streets, enjoy haute cuisine in Michelin-starred restaurants, and cruise along the Seine River. The city's boulevards, gardens, and architecture make it a timeless destination for romance and culture.",
    metadata={"country": "France", "type": "city"}
)

vector_store.update_document(document_id=doc_ids[0], document=updated_doc1)
print(f"Updated document {doc_ids[0]}")

# View updated documents
print("\n=== Documents After Update ===")
all_docs = vector_store.get(include=['documents', 'metadatas'])
print(f"First document now: {all_docs['documents'][0][:150]}...")

# Delete a document
print(f"\n=== Deleting Document {doc_ids[0]} ===")
vector_store.delete(ids=[doc_ids[0]])
print("Document deleted")

# View remaining documents
print("\n=== Remaining Documents ===")
remaining = vector_store.get(include=['documents', 'metadatas'])
print(f"Number of documents: {len(remaining['ids'])}")
for i, (doc_id, doc, meta) in enumerate(zip(remaining['ids'], remaining['documents'], remaining['metadatas'])):
    print(f"\n{i+1}. ID: {doc_id}")
    print(f"   Metadata: {meta}")
    print(f"   Content: {doc[:80]}...")