from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_pdf_files(data_directory):
    loader = DirectoryLoader(
        data_directory, 
         glob="*.pdf", 
         loader_cls=PyPDFLoader
         )
    documents = loader.load()
    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "unknown")}
        )
        minimal_docs.append(minimal_doc)
    return minimal_docs



def text_split(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 20) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
        )
    texts_chunks = text_splitter.split_documents(documents)
    return texts_chunks


def download_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings











