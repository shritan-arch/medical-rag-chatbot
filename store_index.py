import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import download_embeddings, filter_to_minimal_docs, load_pdf_files, text_split


load_dotenv()

DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")


def create_pinecone_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    """Create the Pinecone index if it does not already exist."""
    if pc.has_index(index_name):
        return

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        timeout=60,
    )


def main() -> None:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is required in your environment or .env file.")

    extracted_data = load_pdf_files(str(DATA_DIR))
    minimal_docs = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(minimal_docs)

    embeddings = download_embeddings()
    embedding_dimension = len(embeddings.embed_query("dimension check"))

    pc = Pinecone(api_key=pinecone_api_key)
    create_pinecone_index(pc, INDEX_NAME, embedding_dimension)

    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=pinecone_api_key,
    )

    print(f"Stored {len(text_chunks)} chunks in Pinecone index '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
