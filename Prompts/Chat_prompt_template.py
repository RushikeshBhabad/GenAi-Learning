# Used for Multi turn messages


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
    # Send a tuple in Chat Prompt Template
    ('system', 'You are a helpful {domain} Expert'),
    ('human', 'Explain in simple terms what is {topic}')
])

prompt = chat_template.invoke({'domain':'Cricket Expert', 'topic':'Googly'})

print(prompt)