import os

from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

_ = load_dotenv(find_dotenv())

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file or environment.")

chat_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    max_retries=2,
    api_key=groq_api_key,
)

def load_pdf_to_process(filePath: str) -> str:
        loaded_data = PyPDFLoader(file_path=filePath).load()

        return loaded_data
    
def split_document(filePath: str) -> str:
    page_content = load_pdf_to_process(filePath=filePath)

    text_splitter = CharacterTextSplitter( chunk_size=100, chunk_overlap=10 )
    chunks_to_text = text_splitter.split_documents(documents=page_content)
    
    return chunks_to_text

def generate_embeddings(filePath: str) -> str:
    chunks_to_text = split_document(filePath=filePath)
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_db = FAISS.from_documents(chunks_to_text, embeddings_model)
    retriever = vector_db.as_retriever(search_kwargs={"k":3})

    # response = retriever.invoke("What is the name of the buyer?")
    
    return retriever

def final_runnable(filePath: str) -> str:
    retriever = generate_embeddings(filePath=filePath)

    template = """Answer the question based on the following context
        {context}
        Question: {input}"""

    prompt = ChatPromptTemplate.from_template(template=template)

    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    # response = chain.invoke("Who is the buyer?")
    return chain

def main() -> None:
    print("Upload file for key data extractor:")
    file_path = input("Provide the path to the file: ").strip()
    if not file_path:
        print("Please provide the correct path to the file")
        return None
    
    chain = final_runnable(filePath=file_path)
    print("Ready to ask questions?")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        if not user_input:
            continue
        
        response = chain.invoke(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()

# /Users/raviojha/Downloads/amazon_invoice.pdf
# Who is the buyer?
