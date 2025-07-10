#改写为符合langchain接口的形式,继承BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from typing import List
import requests
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field

class LocalChatGLM3(BaseChatModel):
    base_url: str = Field()
    model: str = Field(default="chatglm3-6b")
    api_key: str = Field(default="")

    def _call_model_with_history(self, prompt: str, history: List[List[str]]) -> tuple[str, List[List[str]]]:
        url = f"{self.base_url.rstrip('/')}/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "prompt": prompt,
            "history": history
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            resp_json = response.json()
            # 取返回的response和history
            return resp_json.get("response", "⚠️ 未返回 response 字段"), resp_json.get("history", history)
        except requests.exceptions.RequestException as e:
            return f"❌ 请求失败：{str(e)}", history

    def _generate(self, messages: List[BaseMessage], stop=None) -> ChatResult:
        history = []
        prompt = ""

        for msg in messages:
            if isinstance(msg, HumanMessage):
                prompt = msg.content
            elif isinstance(msg, AIMessage):
                history.append([prompt, msg.content])
                prompt = ""

        if prompt == "" and history:
            prompt = history[-1][0]

        response, new_history = self._call_model_with_history(prompt, history)
        # 这里可以保存 new_history 用于下一轮对话，比如保存在对象状态或外部
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    @property
    def _llm_type(self) -> str:
        return "local-chatglm3"
