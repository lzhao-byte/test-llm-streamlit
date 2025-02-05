import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import *
from prompts import *
from typing import Annotated, TypedDict

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import InjectedState, ToolNode

# Core invocation of the model
def chatbot(state):
    messages = state["messages"]
    if "llm" not in st.session_state:
        st.session_state['llm'] = "llama3.2"
    selected_llm = st.session_state.llm
    llm = ChatOllama(model=selected_llm)
    response = llm.invoke(messages)
    return {"messages": [response]}# add the response to the messages using LangGraph reducer paradigm


def chatbot_with_tools(state):
    messages = state["messages"]
    if "llm" not in st.session_state:
        st.session_state['llm'] = "llama3.1"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            ("system", search_prompt),
            ("human", "{user_input}"),
        ]
    )
    llm = ChatOllama(model=st.session_state['llm'],
                     temperature=0.5,
                     ).bind_tools([ddg_search_tool])
    chat = prompt | llm
    response = chat.invoke(messages)
    return {"messages": [response]}# add the response to the messages using LangGraph reducer paradigm


def write_chat_history(current_llm):
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
                st.caption(current_llm)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


def setup_graph(with_tools=False):
    if with_tools is False:
        llm_options = ["Llama3.2 (Meta)",  "Phi-4 (Microsoft)", 
                    "Gemma (Google)", "Gemma 2 (Google)", 
                    "DeepSeek (Mini)", "DeepSeek (Small)", "DeepSeek (Medium)"]
        selected_llm = st.sidebar.radio(label="Choose LLM", options=llm_options)

        llm_sel = {
            "Llama3.2 (Meta)": "llama3.2",
            "Llama3.3 (Meta)": "llama3.3",
            "DeepSeek (Mini)": "deepseek-r1:1.5b",
            "DeepSeek (Small)": "deepseek-r1:8b",
            "DeepSeek (Medium)": "deepseek-r1:32b",
            "Gemma (Google)": "gemma:7b",
            "Gemma 2 (Google)": "gemma2:27b",
            "Phi-4 (Microsoft)": "phi4",
            "LlaVA": "llava",
            "BakLLaVA": 'bakllava'
        }
        st.session_state['llm'] = llm_sel.get(selected_llm, 'llama3.2')
    else:
        llm_options = ["Llama3.3 (Meta)", 
                    "Qwen (Alibaba)", 
                    "Mistral"]
        selected_llm = st.sidebar.radio(label="Choose LLM", options=llm_options)

        llm_sel = {
            "Llama3.3 (Meta)": "llama3.3",
            "Mistral": "mistral",
            "Qwen (Alibaba)": "qwen2.5:14b",
        }
        st.session_state['llm'] = llm_sel.get(selected_llm, 'llama3.2')
    st.session_state['graph'] = create_graph(with_tools=with_tools)


# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Custom keys for additional data can be added here such as - conversation_id: str


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "toolNode"
    return END

def create_graph(with_tools=False):
    graph = StateGraph(GraphsState)

    # Define the structure (nodes and directional edges between nodes) of the graph
    graph.add_edge(START, "modelNode")
    if with_tools:
        graph.add_node("modelNode", chatbot_with_tools)
        ddg_search_tool = DuckDuckGoSearchResults(
                        name = "DuckDuckGoSearch",
                        description = "Useful for when you need to answer questions about current events. You should ask targeted questions",
                        backend="news",
                        return_direct=True,
                    )
        graph.add_node("toolNode", ToolNode([ddg_search_tool]))
        graph.add_conditional_edges("modelNode", should_continue, ["toolNode", END])
        graph.add_edge("toolNode", "modelNode")
    else:
        graph.add_node("modelNode", chatbot)
    graph.add_edge("modelNode", END)

    # Compile the state graph into a runnable object
    graph_runnable = graph.compile()

    return graph_runnable

def invoke_graph(graph_runnable, st_messages, callables):
    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    # Invoke the graph with the current messages and callback configuration
    return graph_runnable.invoke({"messages": st_messages}, config={"callbacks": callables})