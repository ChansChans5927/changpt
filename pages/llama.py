import streamlit as st
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.prompts import PromptTemplate
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain import hub

from dotenv import load_dotenv

load_dotenv() 

st.set_page_config(page_title="llama3", page_icon="🦙")
st.title("llama3")

st.sidebar.header("Model Parameters")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
top_p = st.sidebar.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.1)

prompt = hub.pull("x-05/react-chat-history")

# Models
ollama = ChatOllama(
    model="llama3:latest",
    temperature=temperature,
    top_p=top_p
)

search_tool = DuckDuckGoSearchRun(name="Search")

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

tools = [
    search_tool,
    arxiv_tool,
    wiki_tool
]

react_agent = create_react_agent(
    llm=ollama,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True
)

# initial state
if "messages" not in st.session_state:
    st.session_state.messages = []

messages = st.session_state.messages
for msg in messages:
    st.chat_message(msg["role"]).write(msg["content"])

def handle_chat_interaction(user_input):
    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            search_results = agent_executor.invoke(
                {
                    "input": user_input,
                    "chat_history": messages
                }
            )
        st.write(search_results["output"])
    messages.append({"role": "assistant", "content": search_results["output"]})

user_input = st.chat_input("Type your message here...")
if user_input:
    handle_chat_interaction(user_input)
