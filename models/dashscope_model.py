from langchain_openai import ChatOpenAI
from configs import DASHSCOPE_API_KEY
from configs import MODEL_PATH
from configs import EMBED_PATH
from langchain_community.embeddings import DashScopeEmbeddings

'''
def get_llm():
    return ChatOpenAI(
            openai_api_key=DASHSCOPE_API_KEY,
            model=MODEL_PATH,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
        )
'''
def get_embeddings():
    return DashScopeEmbeddings(
            model=EMBED_PATH,
            dashscope_api_key=DASHSCOPE_API_KEY,
            # openai_api_base="https://dashscope.aliyuncs.com/v1"
)