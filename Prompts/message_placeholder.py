from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# chat template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),  # Retrieve and store chat History
    ('human','{query}')
])

chat_history = []
# load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})

print(prompt)

"""
1️⃣ What is a MessagesPlaceholder?

    MessagesPlaceholder is a special placeholder in a chat prompt template that allows you to insert a variable-length list of previous messages (chat history) dynamically when generating the prompt.
    It acts as a “slot” where conversation history or any list of HumanMessage/AIMessage objects can be inserted.
    Unlike a normal {variable}, it does not get replaced as a string, but instead injects message objects directly into the chat sequence.
    Analogy:
    Think of it like a conveyor belt:
    system message → sets context
    MessagesPlaceholder → puts all previous chat messages on the conveyor
    human message → adds the new user query
    So the model sees the entire conversation in order, which is crucial for context-aware responses.

"""