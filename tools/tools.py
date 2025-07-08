from langchain_community.agent_toolkits import load_tools
from langchain_community.tools import DuckDuckGoSearchRun
import models.dashscope_model as model
from agent.rag_qa import build_rag_chain
from langchain.tools import Tool

#def get_tools():
#    tools = load_tools(["llm-math"], llm=model.get_llm())
#    return tools

def get_tools():
    search = DuckDuckGoSearchRun()
    rag_chain = build_rag_chain()

    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=search.run,
            description="搜索互联网信息的工具"
        ),
        Tool(
            name="DocumentQA",
            func=lambda q: rag_chain.run(q),
            description="用于从本地知识文档中回答问题。适合问专业内容、技术资料等。"
        )
    ]
    return tools