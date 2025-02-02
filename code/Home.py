import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


if __name__ == '__main__':

    # Streamlit app title
    st.title('Agent Playground')


    