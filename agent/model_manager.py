from langchain_community.chat_models import ChatOpenAI
from langchain.llms.base import LLM
from langchain_core.language_models import BaseChatModel

from configs import DASHSCOPE_API_KEY
from models.client import LocalChatGLM3


def get_available_models():
    return ["deepseek", "qwen",  "llama", "local"]
#切换模型后history清空
def get_llm(model_name: str) -> BaseChatModel:
    if model_name == "deepseek":
        return ChatOpenAI(
            openai_api_key=DASHSCOPE_API_KEY,
            model="deepseek-r1-distill-llama-70b",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
        )
    elif model_name == "qwen":
        return ChatOpenAI(
            openai_api_key=DASHSCOPE_API_KEY,
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
        )
    elif model_name == "local":
        return LocalChatGLM3(
            base_url="http://localhost:8000",  # 本地部署地址
            api_key="",
            model="chatglm3-6b"
        )
    elif model_name == "llama":
        return ChatOpenAI(
            openai_api_key=DASHSCOPE_API_KEY,
            model="llama-4-maverick-17b-128e-instruct",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
        )
    else:
        raise ValueError(f"不支持的模型: {model_name}")
