# === chains/qa_chain.py ===
from langchain.chains import RetrievalQA
import models.dashscope_model as model
from utils.loader import load_and_split

from langchain.prompts import PromptTemplate

from agent.memory import get_memory

def build_conversational_chain(llm, retriever):
    prompt_template = """你是一个有用的 AI 助手，基于上下文信息回答用户问题。
如果你不知道答案，请说你不知道，避免编造答案。尽量简洁明了。回答结尾加上“谢谢你的提问！”。

{context}

问题: {question}
有用的回答:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=model.get_llm(),
        retriever=retriever,
        memory=get_memory(),
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return chain