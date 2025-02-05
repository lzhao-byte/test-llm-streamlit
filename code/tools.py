from langchain_community.tools import (
    DuckDuckGoSearchResults, 
    WikipediaQueryRun,
    YahooFinanceNewsTool,
    TavilySearchResults
)
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field

from langchain_community.document_loaders import TextLoader, CSVLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool

class WikiInputs(BaseModel):
    """Inputs to the wikipedia tool."""
    query: str = Field(
        description="query to look up in Wikipedia, should be 3 or less words"
    )


def tavily_tool():
    tool = TavilySearchResults(
        max_results = 5,
        include_answer = True,
        include_raw_content = True,
        include_images = False,
        search_depth="advanced",
        description="web search tool"
    )
    return tool

def yahoo_tool():
    tool = YahooFinanceNewsTool()
    return tool

def wiki_tool():
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
    tool = WikipediaQueryRun(
        name="WikiSearch",
        description="look up things in wikipedia about general descriptions and definitions.",
        args_schema=WikiInputs,
        api_wrapper=api_wrapper,
        return_direct=False,
    )
    return tool


class DdgInputs(BaseModel):
    """Inputs to the duckduckgo tool."""
    query: str = Field(
        description="query to search using duckduckgo"
    )

def ddg_search_tool():
    tool = DuckDuckGoSearchResults(
        name = "DuckDuckGoSearch",
        description = "Useful for when you need to answer questions about current events. You should ask targeted questions",
        args_schema=DdgInputs,
        backend="news",
        return_direct=True,
    )
    return tool
