from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
import json

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)


""" 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

def main():
    load_dotenv()  # load .env file

    # Define the model
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-2b-it",
        task="text-generation",
        model_kwargs={"max_new_tokens": 256}  # prevent runaway output
    )
    model = ChatHuggingFace(llm=llm)

    # Output schema
    schema = [
        ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
        ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
        ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
    ]
    parser = StructuredOutputParser.from_response_schemas(schema)

    # Prompt template
    template = PromptTemplate(
        template="Give 3 facts about {topic}\n{format_instruction}",
        input_variables=["topic"],
        partial_variables={"format_instruction": parser.get_format_instructions()},
    )

    # Build the chain
    chain = template | model | parser

    # Run it
    result = chain.invoke({"topic": "black hole"})
    print(result)

if __name__ == "__main__":
    main()


"""