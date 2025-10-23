from langchain.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

# Prompt template with an additional instruction for concise output
summary_prompt = PromptTemplate(
    template=(
        "You are a helpful literary assistant.\n"
        "Write a concise, insightful summary of the following poem:\n\n{poem}"
    ),
    input_variables=["poem"],
)

# Output parser to extract plain text
output_parser = StrOutputParser()

# Load the text document
loader = TextLoader("cricket.txt", encoding="utf-8")
documents = loader.load()

print(f"Type of documents: {type(documents)}")
print(f"Number of documents: {len(documents)}")
print("Sample content:\n", documents[0].page_content[:300], "...")
print("Metadata:", documents[0].metadata)

# Create the chain (Prompt -> Model -> Parser)
chain = summary_prompt | llm | output_parser

# Run the chain on the first document’s content
result = chain.invoke({"poem": documents[0].page_content})
print("\n--- Poem Summary ---\n")
print(result)
