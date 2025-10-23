from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import os

# ---- Load environment variables ----
load_dotenv()

# ---- Hugging Face LLM ----
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
model = ChatHuggingFace(llm=llm)

# ---- Streamlit UI ----
st.set_page_config(page_title="Research Paper Explainer", page_icon="📄", layout="centered")
st.title("📄 Research Paper Explainer")

paper_input = st.selectbox(
    "Select Research Paper:",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

style_input = st.selectbox(
    "Select Explanation Style:",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
)

length_input = st.selectbox(
    "Select Explanation Length:",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"],
)

# ---- Define a reusable PromptTemplate ----
prompt_template = PromptTemplate.from_template(
    """
    You are an expert AI research communicator.

    Summarize the paper "{paper}" 
    in a {length} way using a {style} style.
    Provide a clear, accurate explanation that matches the chosen style and length.
    """
)

if st.button("Summarize"):
    # Fill the template with current user selections
    final_prompt = prompt_template.format(
        paper=paper_input,
        length=length_input,
        style=style_input,
    )

    # Directly call the model with the formatted prompt
    response = model.invoke(final_prompt)

    st.subheader("Generated Summary")
    st.write(response.content)

"""
    1️⃣ Create a JSON Template File
    Make a file named prompt_template.json (you can keep it next to prompt_ui.py).
    Example content:
    {
    "type": "prompt",
    "input_variables": ["paper", "length", "style"],
    "template": "You are an expert AI research communicator.\n\nSummarize the paper \"{paper}\" in a {length} way using a {style} style.\nProvide a clear, accurate explanation that matches the chosen style and length."
    }
    input_variables lists every placeholder you’ll fill.
    template is the actual text with {placeholders}.  
"""