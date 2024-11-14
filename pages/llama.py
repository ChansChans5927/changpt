import streamlit as st
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_core.prompts.prompt import PromptTemplate

from dotenv import load_dotenv

load_dotenv() 

st.set_page_config(page_title="llama3", page_icon="🦙")
st.title("llama3")

st.sidebar.header("Model Parameters")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
top_p = st.sidebar.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.1)

template = """ 
You are a great AI-Assistant that has access to additional tools in order to answer the following questions as best you can. Always answer in the same language as the user question. You have access to the following tools:

{tools}

Chat history:
{chat_history}

To use a tool, please use the following format:

'''
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 3 times)
'''

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
'''
Thought: Do I need to use a tool? No
Final Answer: [your response here]
'''


Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

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

date_tool = Tool.from_function(
    func=lambda x: datetime.now().strftime("%A, %B %d, %Y"),
    name="Current Date",
    description="Useful for when you are need to find the current date and/or time",
)

tools = [
    search_tool,
    arxiv_tool,
    wiki_tool,
    date_tool
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
