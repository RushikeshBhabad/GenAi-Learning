from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF
loader = PyPDFLoader("OOP Unit 5 Notes.pdf")
documents = loader.load()

# Split into smaller chunks
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=""  # keep empty separator as in your code
)

chunks = splitter.split_documents(documents)

# Safely print the second chunk if it exists
if len(chunks) > 1:
    print(chunks[1].page_content)
else:
    print("Not enough chunks to display the second one.")
