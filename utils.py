from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from llm import embeddings


"""
--------------------- UTILITY Functions ---------------------
"""
#? Function -> Loads the text from the pdf provided and splits it into chunks
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

#? Function -> Creates a ChromaDB at given directory and stores the chunks in it
def store_data(chunks, directory, embeddings=embeddings):
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=directory)
    db.persist()

#? Function -> Retrieves the data from the ChromaDB from the given directory
def load_data(directory, embeddings=embeddings):
    vectordb = Chroma(persist_directory=directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k":7})
    return retriever

#? Function -> Asks the question to the LLM via the QA Chain created
def ask_question(qa_chain, retriever, question):
    context = retriever.get_relevant_documents(question)
    answer = (qa_chain(
            {
                "input_documents": context, 
                "context": context,
                "question": question
            },
            return_only_outputs=True
        )
    )['output_text']
    return answer
"""
--------------------- UTILITY Functions ---------------------
"""