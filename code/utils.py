import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage

from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


def write_conversation():
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


def write_chat_history(current_llm):
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
                st.caption(current_llm)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


def setup_llm():
    llm_options = ["Llama3.2 (Meta)", "Llama3.3 (Meta)", "Phi-4 (Microsoft)", 
                   "Gemma (Google)", "DeepSeek (Mini)", "DeepSeek (Small)", "DeepSeek (Medium)"]
                #    "LlaVA (Multimodel)"]#, "Llama3.3 (Meta)",  "DeepSeek (Small)", "DeepSeek (Medium)"]
    selected_llm = st.sidebar.radio(label="Choose LLM", options=llm_options)

    llm_sel = {
        "Llama3.2 (Meta)": "llama3.2",
        "Llama3.3": "llama3.3",
        "DeepSeek (Mini)": "deepseek-r1:1.5b",
        "DeepSeek (Small)": "deepseek-r1:7b",
        "DeepSeek (Medium)": "deepseek-r1:32b",
        "Gemma (Google)": "gemma:7b",
        "Phi-4 (Microsoft)": "phi4",
        "LlaVA": "llava",
        "BakLLaVA": 'bakllava'
    }
    st.session_state['llm'] = llm_sel.get(selected_llm, 'llama3.2')