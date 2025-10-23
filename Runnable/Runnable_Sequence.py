from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel

load_dotenv()

# ---------- Prompts ----------
# Step 1: Generate a concise, witty joke
joke_prompt = PromptTemplate(
    template=(
        "You are a clever comedian. "
        "Write a short, original joke about the topic: {topic}. "
        "Make it witty but family-friendly."
    ),
    input_variables=["topic"],
)

# Step 2: Explain the joke’s meaning and the wordplay
explain_prompt = PromptTemplate(
    template=(
        "Explain why the following joke is funny, including any wordplay or cultural references:\n\n"
        "{joke_text}"
    ),
    input_variables=["joke_text"],
)

# ---------- Model ----------
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=120,
    temperature=0.7,
    top_p=0.9,
)

chat_model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# ---------- Chain ----------
# Wrap the output of the first parser so it feeds as {joke_text} to the second prompt
format_for_explanation = RunnableLambda(lambda text: {"joke_text": text})

chain = RunnableSequence(
    joke_prompt,
    chat_model,
    parser,
    format_for_explanation,
    explain_prompt,
    chat_model,
    parser,
)

# ---------- Run ----------
if __name__ == "__main__":
    result = chain.invoke({"topic": "artificial intelligence"})
    print("---- Joke and Explanation ----")
    print(result)
