import os

from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_embeddings
from src.prompt import system_prompt


load_dotenv()

app = Flask(__name__)

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")


def check_environment() -> str | None:
    """Return an error message when required environment variables are missing."""
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        return f"Missing required environment variable(s): {', '.join(missing_vars)}"

    return None


def create_rag_chain():
    embeddings = download_embeddings()

    docsearch = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    chat_model = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(
        llm=chat_model,
        prompt=prompt,
    )

    return create_retrieval_chain(retriever, document_chain)


rag_chain = None
startup_error = check_environment()

if startup_error is None:
    try:
        rag_chain = create_rag_chain()
    except Exception as exc:
        startup_error = f"Could not initialize the chatbot: {exc}"


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    question = ""
    error = startup_error

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if not question:
            error = "Please enter a question."
        elif rag_chain is None:
            error = startup_error or "The chatbot is not ready. Please check your setup."
        else:
            try:
                response = rag_chain.invoke({"input": question})
                answer = response.get("answer", "I could not find an answer.")
                error = None
            except Exception as exc:
                error = f"Something went wrong while answering: {exc}"

    return render_template(
        "index.html",
        answer=answer,
        question=question,
        error=error,
        index_name=PINECONE_INDEX_NAME,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
