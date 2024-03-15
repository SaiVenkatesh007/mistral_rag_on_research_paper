"""
File: app.py
Description: The main code file for the Streamlit Application
"""

#* Import Statements
import streamlit as st

from llm import load_llm, create_qa_chain
from utils import get_text, store_data, load_data, ask_question

#* Loading the LLM
llm = load_llm()


"""
--------------------- Streamlit Application ---------------------
"""
st.title("Research Paper QA Bot")

#* Section for uploading (Via File or PDF Link)
pdf_file = st.file_uploader("Upload PDF", type=['pdf'])
st.write("<h5 style='text-align:center;color:white;'>OR</h5>", unsafe_allow_html=True)
pdf_link = st.text_input("PDF Link")

#* Button to Continue after providing the PDF
if st.button("Confirm"):

    #* Checking whether we received a PDF or not
    if pdf_file is not None or pdf_link != "":
        st.write("PDF Received succesfully!")

        #* File or URL check
        if pdf_file is not None:
            #* Creating a buffer file to get the text from PDF
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.getbuffer())
            chunks = get_text("temp.pdf")
        else:
            #* Getting the text from PDF URL
            chunks = get_text(pdf_link)

        #print("Started Storing Data\n") # Debug
        #* Storing the Data into a ChromaDB
        store_data(chunks=chunks, directory="data") #! "data" is the directory of the DB
        #! Directory can be changed upon need
        # print("Successfully stored Data\n\nStarted Retrieving Data\n") # Debug

        #* Loading the data from the DB
        retriever = load_data("data")
        # print("Successfully Retrieved Data\nStarted Creating a QA Chain\n") # Debug

        #* Creating a QA Chain
        qa_chain = create_qa_chain(llm=llm)
        # print("Successfully Created a QA Chain\n") # Debug

        #* Section for Asking a Question
        question = st.text_input("Ask a question: ")
        if st.button("Get Answer"):
            # print("Reached After Clicking the Button") # Debug

            #* Sending the question to the LLM
            answer = ask_question(qa_chain=qa_chain, retriever=retriever, question=question)
            # print(answer) # Debug
            # print("Tried printing the answer") # Debug

            #* Displaying the Answer from the LLM
            st.write("Answer:")
            st.write(answer)
    else:
        #* In case we don't receive any PDF
        st.write("Did not receive any PDF!")
"""
--------------------- Streamlit Application ---------------------
"""