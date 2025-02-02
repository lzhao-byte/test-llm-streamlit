from utils import *
from streaming import *
from graph import *


def main():
    setup_llm()
    current_llm = f"Currently using {st.session_state['llm']}"

    ## clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.pop("chat_history", None)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am your assistant. How can I help?")
        ]
        
    write_chat_history(current_llm)

    # takes new input in chat box from user and invokes the graph
    if user_input := st.chat_input("Type your message here..."):
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.chat_message("Human").write(user_input)

        # Process the AI's response and handles graph events using the callback mechanism
        with st.chat_message("AI"):
            # create a new container for streaming messages only, and give it context
            container = st.empty()
            st_callback = StreamHandler(container)
            response = invoke_graph(st.session_state.chat_history, [st_callback])
            st.caption(current_llm)

            # Add that last message to the st_message_state
            # Streamlit's refresh the message will automatically be visually rendered bc of the msg render for loop above
            completion = re.sub("<think>(.*[\s\S]*)<\/think>", "", response["messages"][-1].content)
            st.session_state.chat_history.append(AIMessage(content=completion))


if __name__ == "__main__":
    st.set_page_config(page_title="Chatbot")
    st.markdown("### Simple Chatbot")
    main()