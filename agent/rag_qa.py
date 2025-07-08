# agent/rag_qa.py
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import psutil

from models.dashscope_model import get_embeddings
import os

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

def load_knowledge_base(kb_name: str):
    if not kb_name or not isinstance(kb_name, str):
        kb_name = "default_kb"  # 强制设置默认值
    kb_path = os.path.join(VECTORSTORE_DIR, kb_name)
    return Chroma(persist_directory=kb_path, embedding_function=get_embeddings()).as_retriever()

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

"""
def build_rag_chain():
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    #embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(chunks, embedding=get_embeddings(), persist_directory=".chroma")

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=get_llm(), retriever=retriever, return_source_documents=False)
    return qa_chain
"""
