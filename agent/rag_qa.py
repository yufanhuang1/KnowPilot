# agent/rag_qa.py
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from models.dashscope_model import get_embeddings,get_llm
import os
# 文档路径（可支持 txt、md、pdf 需插件）
#DOC_PATH = "data/Build a Large Language Model.pdf"
DOC_PATH = "C:/Users/vennhuang/Desktop/text.txt"

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

