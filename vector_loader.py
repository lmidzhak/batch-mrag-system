from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import faiss
import joblib
import os


def load_text_vectorstore():
    embedder = OpenAIEmbeddings()
    return FAISS.load_local(
        "vectorstore/faiss_openai",
        embeddings=embedder,
        allow_dangerous_deserialization=True
    )


def load_image_vectorstore():
    index_path = "vectorstore/faiss_images.index"
    metadata_path = "vectorstore/image_metadata.pkl"

    if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
        raise FileNotFoundError("Image vectorstore files are missing.")

    index = faiss.read_index(index_path)
    metadata = joblib.load(metadata_path)

    return index, metadata
