from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = '/home/bumblebee/huggingface_cache'

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"  # Smaller & optimized model

llm = HuggingFacePipeline.from_model_id(
    model_id=MODEL_ID,
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result.content)
