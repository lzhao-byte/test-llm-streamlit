from langchain_core.callbacks import BaseCallbackHandler
import re
import streamlit as st

# Define a custom callback handler class for managing and displaying stream events from LangGraph in Streamlit
class StreamHandler(BaseCallbackHandler):
    """
    Custom callback handler for Streamlit that updates a Streamlit container with new tokens.
    """

    def __init__(self, container, initial_text: str = ""):
        """
        Initializes the StreamHandler with a Streamlit container and optional initial text.

        Args:
            container (DeltaGenerator): The Streamlit container where text will be rendered.
            initial_text (str): Optional initial text to start with in the container.
        """
        self.container = container  # The Streamlit container to update
        self.text = initial_text  # Initialize the text content, starting with any initial text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Callback method triggered when a new token is received (e.g., from a language model).

        Args:
            token (str): The new token received.
            **kwargs: Additional keyword arguments.
        """
        self.text += token  # Append the new token to the existing text
        if '</think>' in self.text:
            think_process = re.match("<think>(.*[\s\S]*)<\/think>(.*[\s\S]*)", self.text)
        else:
            think_process = re.match("<think>(.*[\s\S]*)()", self.text)

        if think_process is not None:
            thoughts, completion = think_process.groups()
            if thoughts:
                with self.container.container():
                    with st.expander("Thought Process"):
                        st.caption(thoughts.strip())
                    st.write(completion.strip())
        else:
            self.container.container().write(self.text)
