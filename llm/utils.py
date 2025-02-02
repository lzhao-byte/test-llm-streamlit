import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage

from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


def record_chat(llm):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hellow, I am a chatbot. How can I help?")
        ]
    write_conversation()
    user_input = st.chat_input("Type your message here...")
    if user_input is not None and user_input != "":
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("Human"):
            st.markdown(user_input)
        with st.chat_message("AI"):
            response = st.write_stream(generate_response(llm, user_input, st.session_state.chat_history))
        st.session_state.chat_history.append(AIMessage(content=response))


def write_conversation():
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


def generate_response(llm, user_input, chat_history):
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_input}
    """
    prompt = ChatPromptTemplate.from_template(template)
    memory = ConversationBufferMemory()
    outputparser = StrOutputParser()
    chain = prompt | llm |outputparser
    result = chain.stream(
        {"user_input": user_input},
        {"chat_history": chat_history}
    )
    return result


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