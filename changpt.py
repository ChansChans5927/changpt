import sys
import streamlit as st
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv

# 환경 변수 로드 및 템플릿 파일 경로 추가
load_dotenv()
sys.path.append("./chat_prompt_templete.py")
from chat_prompt_templete import get_prompt_template

# Streamlit 앱 기본 설정
st.set_page_config(page_title="Chan-GPT", page_icon="🤖")
st.title("Chan-GPT")
st.write("Chat with Chan-GPT to get stock recommendation and investment duration.")

#모델 파라미터를 사이드바에서 초기화
def initialize_parameters():
    st.sidebar.header("Model Parameters")
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    top_p = st.sidebar.slider("Top-p", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    return temperature, top_p

#에이전트가 사용할 도구들을 생성하고 반환
def create_tools():
    # Google Serper API Wrapper 설정
    search_tool = GoogleSerperAPIWrapper()
    
    # Tool 객체로 래핑
    search_tool = Tool(
        name="Google Search",
        func=search_tool.run,
        description="Use this tool to search for information on the web using Google Search."
    )
    
    date_tool = Tool.from_function(
        func=lambda x: datetime.now().strftime("%A, %B %d, %Y"),
        name="Current Date",
        description="Useful for when you are need to find the current date and/or time",
    )
    
    return [search_tool, date_tool]

#에이전트 실행자를 생성하고 반환
def create_agent_executor(tools, temperature, top_p):
    llama = ChatOllama(
        model="changpt:latest",
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

# 검수 AI
def create_review_agent():
    reviewer_llm = ChatOllama(
        model="fixer:latest"
    )
    return reviewer_llm

#세션 상태에 저장된 채팅 기록을 화면에 표시
def display_chat_history():
    messages = st.session_state.get("messages", [])
    for msg in messages:
        st.chat_message(msg["role"]).write(msg["content"])
    return messages

#사용자의 입력을 처리하고 assistant의 응답을 표시
def handle_chat_interaction(user_input, agent_executor, reviewer, messages):
    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            search_results = agent_executor.invoke(
                {"input": user_input, "chat_history": messages}
            )
        output_text = search_results["output"]
        # 검수
        review_prompt = f"Please review the following response: {output_text}"
        review_result = reviewer.invoke(review_prompt).content.strip("'''")
        st.write(review_result)
        
    messages.append({"role": "assistant", "content": output_text})
    st.session_state.messages = messages

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 파라미터 초기화
temperature, top_p = initialize_parameters()
# 에이전트 도구 생성
tools = create_tools()
# 에이전트 실행자 생성
agent_executor = create_agent_executor(tools, temperature, top_p)
reviewer = create_review_agent()

# 채팅 기록 표시
messages = display_chat_history()

# 사용자의 입력 처리
user_input = st.chat_input("Type your message here...")
if user_input:
    handle_chat_interaction(user_input, agent_executor, reviewer, messages)