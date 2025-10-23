from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
model = ChatHuggingFace(llm=llm)

messages =[
    SystemMessage(content = 'You are a data Scientist'),
    HumanMessage(content = 'Tell me about complex data preprocessing steps addvance level.')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)