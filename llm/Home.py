import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


def get_response(llm_choice):
    # Define output parser
    output_parser = StrOutputParser()
    llm_sel = {
        "Llama3.2": "llama3.2",
        "Llama3.3": "llama3.3",
        "DeepSeek (Mini)": "deepseek-r1:1.5b",
        "DeepSeek (Small)": "deepseek-r1:7b",
        "DeepSeek (Medium)": "deepseek-r1:32b",
        "Gemma": "gemma:7b",
        "Phi-4": "phi4",
        "LlaVA": "llava",
        "BakLLaVA": 'bakllava'
    }
    llm = ChatOllama(model=llm_sel.get(llm_choice, 'llama3.2'))
    
    # Build Langchain pipeline
    chain = prompt | llm | output_parser

    # Generate response if user input exists
    if input_text:
        return chain.invoke({'question': input_text})
    else:
        return None

def main():
    # User input field
    input_text = st.text_input("Search the topic you want")

    # Select LLM option (can be extended for more choices)
    llm_options = ["Llama3.2 (Meta)", "Phi-4 (Microsoft)", "Gemma (Google)", "DeepSeek (Mini)", "LlaVA (Multimodel)"]#, "Llama3.3 (Meta)",  "DeepSeek (Small)", "DeepSeek (Medium)"]
    selected_llm = st.selectbox("Choose LLM", llm_options)

    # Generate response based on selection
    response = get_response(selected_llm)

    # Display response if available
    if response:
        st.write(f"**Response from {selected_llm}:**")
        st.write(response)

if __name__ == '__main__':
 
    # Prompt template defining conversation flow
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question: {question}")
    ])

    # Streamlit app title
    st.title('Agent Playground')


    