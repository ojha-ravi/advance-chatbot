import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain

_ = load_dotenv(find_dotenv())

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

groq_api_key = os.environ["GROQ_API_KEY"]

if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file or environment.")

chat_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    max_retries=2,
    api_key=groq_api_key,
)

base_dir = Path(__file__).resolve().parent
sql_db_path = base_dir / "data" / "street_tree_db.sqlite"

db = SQLDatabase.from_uri(f"sqlite:///{sql_db_path}")

query_chain = create_sql_query_chain(chat_model, db)

def answer_question(question: str) -> str:
    generated_sql = query_chain.invoke({"question": question})
    return db.run(generated_sql)

def main() -> None:
    print("Ask a question about the street tree database. Type 'exit' to quit.")
    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print(answer_question(question))
    
if __name__ == "__main__":
    main()
