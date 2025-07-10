import streamlit as st
from langchain.chains.retrieval_qa.base import RetrievalQA
from agent.agent_chain import CustomAgentExecutor
from agent.model_manager import get_available_models, get_llm
from agent.rag_qa import list_knowledge_bases, ingest_document, delete_knowledge_base, load_knowledge_base


st.set_page_config(page_title="Agent Chatbot", layout="wide")
st.title("🤖 chatbot 🤖")

st.sidebar.header("🧠 模型选择")
model_list = get_available_models()
selected_model = st.sidebar.selectbox("选择模型", model_list)

# 用选择的模型初始化 LLM
llm = get_llm(selected_model)

# --- 上传文档 ---
st.sidebar.header("📤 上传文档")
uploaded_file = st.sidebar.file_uploader("选择文档", type=["txt", "pdf", "md"])
kb_name = st.sidebar.text_input("知识库名称", value="default")

if uploaded_file and st.sidebar.button("上传并入库"):
    ingest_document(uploaded_file, kb_name)
    st.sidebar.success("文档已入库！请刷新页面以更新知识库工具")

# --- 删除知识库 ---
#st.sidebar.header("🗑 删除知识库")
#kb_to_delete = st.sidebar.selectbox("选择要删除的知识库", list_knowledge_bases())
#if st.sidebar.button("删除知识库"):
#    delete_knowledge_base(kb_to_delete)
#    st.sidebar.warning(f"已删除知识库 {kb_to_delete}")

# --- 展示当前知识库 ---
st.sidebar.header("📂 当前知识库")
st.sidebar.write(", ".join(list_knowledge_bases()) or "暂无知识库")

if "agent" not in st.session_state or st.session_state.selected_model != selected_model:

    st.session_state.selected_model = selected_model
    st.session_state.agent = CustomAgentExecutor(llm=llm)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("请输入你的问题...")

if user_input:
    with st.spinner("🤖 正在思考..."):
        response = st.session_state.agent.run( user_input)

    # 如果返回的是 dict，取出 output 字段；否则直接用原样
    final_output = response.get("output") if isinstance(response, dict) else response
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("ai", final_output))

for role, msg in st.session_state.chat_history:
    with st.chat_message("🤖" if role == "ai" else "🧑"):
        st.markdown(msg)



#streamlit run app.py
#localhost:8501

