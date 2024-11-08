import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="ChanGPT", page_icon="🤖")
st.title("ChanGPT")

st.sidebar.header("Model Parameters")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
top_p = st.sidebar.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.1)

# Models
llm = ChatOllama(model="changpt:latest", temperature=temperature, top_p=top_p)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Chan-GPT. When a user asks for a stock recommendation, suggest a high-growth company from the S&P 500 index."
               "If there are multiple candidates, recommend only one stock."
               "Also, inform the user about the suggested investment duration in terms of months."),
    ("user", "{input}")
])

# Chaining
chain = prompt | llm | StrOutputParser()

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
    response_stream = chain.stream(input=user_input)
    with st.chat_message("assistant"):
        response = st.write_stream(response_stream)
    messages.append({"role": "assistant", "content": response})

# Chat Input
user_input = st.chat_input("Type your message here...")
if user_input:
    handle_chat_interaction(user_input)