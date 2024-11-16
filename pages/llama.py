import sys
import streamlit as st
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun
from dotenv import load_dotenv

# 환경 변수 로드 및 템플릿 파일 경로 추가
load_dotenv()
sys.path.append("../chat_prompt_templete.py")
from chat_prompt_templete import get_prompt_template

# Streamlit 앱 기본 설정
st.set_page_config(page_title="llama3", page_icon="🦙")
st.title("llama3")

#모델 파라미터를 사이드바에서 초기화
def initialize_parameters():
    st.sidebar.header("Model Parameters")
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    top_p = st.sidebar.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
    return temperature, top_p

#에이전트가 사용할 도구들을 생성하고 반환
def create_tools():
    search_tool = DuckDuckGoSearchRun(name="Search")
    
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    
    date_tool = Tool.from_function(
        func=lambda x: datetime.now().strftime("%A, %B %d, %Y"),
        name="Current Date",
        description="Useful for when you are need to find the current date and/or time",
    )

    google_serper_tool = Tool.from_function(
        func=GoogleSerperAPIWrapper().run,
        name="Google Search",
        description="Search using Google Serper API to find precise information"
    )
    
    return [search_tool, arxiv_tool, google_serper_tool, date_tool]

#에이전트 실행자를 생성하고 반환
def create_agent_executor(tools, temperature, top_p):
    llama = ChatOllama(
        model="llama3:latest",
        temperature=temperature,
        top_p=top_p
    )
    
    react_agent = create_react_agent(
        llm=llama,
        tools=tools,
        prompt=get_prompt_template()
    )
    
    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True
    )

#세션 상태에 저장된 채팅 기록을 화면에 표시
def display_chat_history():
    messages = st.session_state.get("ollama_messages", [])
    for msg in messages:
        st.chat_message(msg["role"]).write(msg["content"])
    return messages

#사용자의 입력을 처리하고 assistant의 응답을 표시
def handle_chat_interaction(user_input, agent_executor, messages):
    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            search_results = agent_executor.invoke(
                {"input": user_input, "chat_history": messages}
            )
        st.write(search_results["output"])
        
    messages.append({"role": "assistant", "content": search_results["output"]})
    st.session_state.ollama_messages = messages

# 세션 상태 초기화
if "ollama_messages" not in st.session_state:
    st.session_state.ollama_messages = []

# 파라미터 초기화
temperature, top_p = initialize_parameters()
# 에이전트 도구 생성
tools = create_tools()
# 에이전트 실행자 생성
agent_executor = create_agent_executor(tools, temperature, top_p)

# 채팅 기록 표시
messages = display_chat_history()

# 사용자의 입력 처리
user_input = st.chat_input("Type your message here...")
if user_input:
    handle_chat_interaction(user_input, agent_executor, messages)