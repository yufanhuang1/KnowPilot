from langchain.vectorstores import Chroma
from langchain.embeddings import DashScopeEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_vectordb():
    loader = TextLoader("./data/docs/yourfile.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = DashScopeEmbeddings()
    vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory="./vectorstore/chroma_db")
    return vectordb
