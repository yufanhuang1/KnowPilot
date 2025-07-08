import streamlit as st
#from agent.agent_chain import create_agent
from agent.agent_chain import CustomAgentExecutor

st.set_page_config(page_title="Agent Chatbot", layout="wide")
st.title("🤖 chatbot 🤖")

if "agent" not in st.session_state:
    #st.session_state.agent = create_agent()
    st.session_state.agent = CustomAgentExecutor()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("请输入你的问题...")

if user_input:
    with st.spinner("🤖 正在思考..."):
        response = st.session_state.agent.run( user_input)

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("ai", response))

for role, msg in st.session_state.chat_history:
    with st.chat_message("🤖" if role == "ai" else "🧑"):
        st.markdown(msg)



#streamlit run app.py
#localhost:8501

