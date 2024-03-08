from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from llm import model_kwargs, embeddings

def get_text(pdf):
    loader = PyPDFLoader(pdf, extract_images=True)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 4000,
        chunk_overlap = 200,
        length_function = len,
        add_start_index = True,
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

def store_data(chunks, directory, embeddings=embeddings):
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=directory)
    db.persist()

def load_data(directory, embeddings=embeddings):
    vectordb = Chroma(persist_directory=directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k":7})
    return retriever
