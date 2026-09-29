import os
from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

_= load_dotenv(find_dotenv())

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file or environment.")

class Classification(BaseModel):
    sentiment: str = Field(description="Sentiment of the text")
    political_tendency: str = Field(description="The Political tendency of the user")
    language: str = Field(description="Language of the text")

chat_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    max_retries=2,
    api_key=groq_api_key,
).with_structured_output(schema=Classification)

template = """Extract the desired information from the following passage.
        Only extract properties mentioned in the Classification schema
        Passage: {input}"""

prompt = ChatPromptTemplate.from_template(template=template)

chain = prompt | chat_model

def main() -> None:
    print("Bot is ready to understand the semantics of the text. Enter the text/comment.")
    user_input = input("You: ").strip()
    response = chain.invoke(user_input)
    print(response)
    
if __name__ == "__main__":
    main()
