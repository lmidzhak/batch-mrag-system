from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def load_vectorstore():
    embedder = OpenAIEmbeddings()
    return FAISS.load_local(
        "vectorstore/faiss_openai",
        embeddings=embedder,
        allow_dangerous_deserialization=True
    )
