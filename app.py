import streamlit as st
from utils import get_text, store_data

st.title("Research Paper QA Bot")

pdf_file = st.file_uploader("Upload PDF", type=['pdf'])
st.write("<h5 style='text-align:center;color:white;'>OR</h5>", unsafe_allow_html=True)
pdf_link = st.text_input("PDF Link")

if st.button("Confirm"):
    if pdf_file is not None or pdf_link != "":
        st.write("PDF Received succesfully!")
        if pdf_file is not None:
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.getbuffer())
            chunks = get_text("temp.pdf")
        else:
            chunks = get_text(pdf_link)
        store_data(chunks=chunks, directory="data")
    else:
        st.write("Did not receive any PDF!")