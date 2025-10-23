from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found! Please add GOOGLE_API_KEY to your .env file.")

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.6,
    max_output_tokens=200,
    google_api_key=api_key
)

# Ask the first question
question1 = "What is the capital of India?"
result1 = model.invoke([{"role": "user", "content": question1}]) 

# Ask a custom question
question2 = input("Enter your question: ")
result2 = model.invoke([{"role": "user", "content": question2}])  

# Print results
print("\n===== GEMINI RESPONSES =====")
print(f"Q1: {question1}")
print(f"A1: {result1.content}\n")
print(f"Q2: {question2}")
print(f"A2: {result2.content}\n")
