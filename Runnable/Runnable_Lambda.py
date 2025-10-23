from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnableParallel,
)

load_dotenv()

# ------------------ Model ------------------
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=120,
    temperature=0.7,
    top_p=0.9,
)
model = ChatHuggingFace(llm=llm)

# ------------------ Helper ------------------
def word_count(text: str) -> int:
    """Return number of words in the text."""
    return len(text.split())

# ------------------ Prompts -----------------
joke_prompt = PromptTemplate(
    template=(
        "You are a witty, family-friendly comedian.\n"
        "Write a short, original joke about the topic: {topic}"
    ),
    input_variables=["topic"],
)

parser = StrOutputParser()

# ------------------ Chains ------------------
# 1️⃣ Generate joke text
joke_gen_chain = RunnableSequence(joke_prompt, model, parser)

# 2️⃣ Parallel branch:
#     a. Pass through the joke as-is
#     b. Count the words
parallel_chain = RunnableParallel({
    "joke": RunnableLambda(lambda x: x),      # passthrough
    "word_count": RunnableLambda(word_count), # word count
})

# 3️⃣ Combine them into a final sequence
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# ------------------ Run ---------------------
if __name__ == "__main__":
    result = final_chain.invoke({"topic": "artificial intelligence"})
    # Format output
    output = f"{result['joke']}\nWord count: {result['word_count']}"
    print(output)
