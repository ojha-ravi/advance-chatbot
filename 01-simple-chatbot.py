import os

from dotenv import find_dotenv, load_dotenv
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

_ = load_dotenv(find_dotenv())

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file or environment.")

chat_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    max_retries=2,
    api_key=groq_api_key,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

def get_chat_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(f"chats/{session_id}.json")

chain = RunnableWithMessageHistory(
    prompt | chat_model,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def main() -> None:
    print("Chatbot ready. Type 'exit' to quit.")
    print("*"*10)
    print("Tell me your name to start.")
    user_name = input("Enter you name: ").strip()

    if not user_name:
        print("Please provide your name to start, GoodBye!!")
        return None

    response = chain.invoke(
                {"input": f'User name is {user_name}'},
                config={"configurable": {"session_id": user_name}},
            )

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": user_name}},
        )
        print(f"Bot: {response.content}")

if __name__ == "__main__":
    main()
