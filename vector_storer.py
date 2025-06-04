from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from parser import load_documents
from dotenv import load_dotenv
import os

load_dotenv()


def build_vectorstore():
    documents = load_documents()
    embedder = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embedding=embedder)
    vectorstore.save_local("vectorstore/faiss_openai")
    print("✅ Vectorstore saved to vectorstore/faiss_openai")


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"]
    build_vectorstore()
