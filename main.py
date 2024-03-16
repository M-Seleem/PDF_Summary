import os
import streamlit as st
from utils import *


def main():
    st.set_page_config(page_title="PDF Summarizer")
    st.title("PDF Summarizer APP")
    st.write("Too Lazy To Read? Summarize Your PDF In Just A Few Seconds!")
    st.divider()

    pdf = st.file_uploader("Upload your PDF document", type='pdf')
    submit = st.button("Generate Summary")

    os.environ["OPENAI_API_KEY"] = ""

    if submit:
        response = summarizer(pdf)
        st.subheader("Summary of file:")
        st.write(response)




if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
