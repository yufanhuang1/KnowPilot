# agent/rag_qa.py
import pickle

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import psutil
from models.dashscope_model import get_embeddings
import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient

from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from chromadb import Client
from chromadb.config import Settings

BASE_DOC_DIR = "./data/docs"
VECTORSTORE_DIR = "./data/vectorstores"

def get_loader(file_path: str):
    if file_path.endswith(".txt"):
        return TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        return PyMuPDFLoader(file_path)
    elif file_path.endswith(".md"):
        return UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError("不支持的文件类型")

def list_knowledge_bases():
    return [d for d in os.listdir(VECTORSTORE_DIR) if os.path.isdir(os.path.join(VECTORSTORE_DIR, d))]
"""
#构建向量数据库并持久化
def ingest_document(file, kb_name: str):
    kb_path = os.path.join(VECTORSTORE_DIR, kb_name)
    os.makedirs(kb_path, exist_ok=True)

    file_path = os.path.join(BASE_DOC_DIR, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())

    loader = get_loader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(splits, embedding=get_embeddings(), persist_directory=kb_path)
    vectordb.persist()
"""
def ingest_document(file_path: str, kb_name: str):
    """
    使用 Chroma 将文档嵌入后存入本地知识库
    :param file_path: 本地文档路径，如 docs/my_doc.pdf
    :param kb_name: 知识库名称（Chroma collection 名）
    """

    # 路径准备
    kb_path = os.path.join(VECTORSTORE_DIR, kb_name)
    os.makedirs(kb_path, exist_ok=True)

    # 加载文档
    loader = get_loader(file_path)
    docs = loader.load()

    # 切分文本
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    splits = splitter.split_documents(docs)

    # 构建 Chroma 向量库
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        persist_directory=kb_path,
        collection_name=kb_name,
    )
    # 持久化向量库
    vector_store.persist()

    doc_cache_path = os.path.join(kb_path, "raw_docs.pkl")
    with open(doc_cache_path, "wb") as f:
        pickle.dump(splits, f)

    print(f"[✓] 文档已成功导入知识库 '{kb_name}'，存储路径: {kb_path}")
    print(f"[✓] 原始文档切片已保存到: {doc_cache_path}，共 {len(splits)} 条")

# 加载纯向量检索器
def load_knowledge_base(kb_name: str):
    if not kb_name or not isinstance(kb_name, str):
        kb_name = "default_kb"  # 强制设置默认值
    kb_path = os.path.join(VECTORSTORE_DIR, kb_name)
    return Chroma(persist_directory=kb_path, embedding_function=get_embeddings())

from langchain.schema import BaseRetriever
from typing import List
from pydantic import PrivateAttr
# 加载混合检索器
def load_hybrid_retriever(kb_name: str):
    kb_path = os.path.join(VECTORSTORE_DIR, kb_name)

    # 获取向量数据库
    vector_store = load_knowledge_base(kb_name)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    doc_cache_path = os.path.join(kb_path, "raw_docs.pkl")
    if not os.path.exists(doc_cache_path):
        raise ValueError(f"[{kb_name}] 的原始文档 raw_docs.pkl 不存在，请先执行 ingest_document() 导入文档。")

    with open(doc_cache_path, "rb") as f:
        docs = pickle.load(f)

    if not docs:
        raise ValueError(f"[{kb_name}] 的原始文档为空，无法构建 BM25Retriever。")

    # 提取文档用于 BM25 构建
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    def hybrid_retrieve(query: str) -> list[Document]:
        bm25_docs = bm25_retriever.get_relevant_documents(query)
        vector_docs = vector_retriever.get_relevant_documents(query)

        # 去重合并（即“召回融合”）  对比得分融合
        #分别召回 BM25 和 向量 检索的文档；
        #把两个结果合并在一起；
        #去掉重复文档（用 page_content 做唯一性判断）
        seen = set()
        merged = []
        for doc in bm25_docs + vector_docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        return merged

    # 返回一个兼容 LangChain 的“Retriever”对象
    class HybridRetriever(BaseRetriever):
        _retrieve_func: any = PrivateAttr()  # ✅ 声明私有属性

        def __init__(self, retrieve_func):
            super().__init__()
            self._retrieve_func = retrieve_func

        def get_relevant_documents(self, query: str) -> List[Document]:
            return self._retrieve_func(query)

        async def aget_relevant_documents(self, query: str) -> List[Document]:
            return self.get_relevant_documents(query)

    return HybridRetriever(hybrid_retrieve)

# 用于 QA Chain
def load_retriever(kb_name: str, mode: str = "hybrid"):
    if mode == "vector":
        return load_knowledge_base(kb_name).as_retriever()
    elif mode == "hybrid":
        return load_hybrid_retriever(kb_name)
    else:
        raise ValueError(f"Unsupported retriever mode: {mode}")

def kill_process_using_file(file_path):
    killed=False
    for proc in psutil.process_iter(['pid', 'name', 'open_files']):
        try:
            open_files = proc.info.get('open_files', []) or []  # 处理 None 情况
            for fl in open_files:
                # 使用 os.path.realpath 解析符号链接和绝对路径
                target_path = os.path.realpath(file_path)
                if os.path.realpath(fl.path) == target_path:
                    proc.kill()
                    killed = True  # 标记已终止至少一个进程
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return killed

def delete_knowledge_base(kb_name: str):
    import shutil
    kb_path = os.path.join(VECTORSTORE_DIR, kb_name)
    #if kill_process_using_file(kb_path):
        #shutil.rmtree(os.path.dirname(kb_path))
    if not os.path.exists(kb_path):
        return
        # 先释放数据库资源
    kill_process_using_file(kb_path)  # 终止占用进程
    #safe_rmtree(kb_path)  # 安全删除目录

