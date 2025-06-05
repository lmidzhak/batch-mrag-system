import os
import faiss
import numpy as np
import joblib
import requests

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from parser import load_documents
from text_embedder import get_image_embedding
from dotenv import load_dotenv

from PIL import Image
from io import BytesIO
from tqdm import tqdm

load_dotenv()
os.makedirs("vectorstore", exist_ok=True)


def build_text_vectorstore(documents):

    embedder = OpenAIEmbeddings()
    try:
        vectorstore = FAISS.from_documents(documents, embedding=embedder)
        vectorstore.save_local("vectorstore/faiss_openai")
        print("✅ Text vectorstore saved to vectorstore/faiss_openai")
    except Exception as e:
        print(f"❌ Failed to create text vectorstore: {e}")


def build_image_vectorstore(documents):
    image_vectors = []
    image_metadata = []
    seen_urls = set()

    all_images = [
        (doc, img)
        for doc in documents
        for img in doc.metadata.get("image_urls", [])
        if isinstance(img, dict)
    ]

    for doc, img in tqdm(all_images, desc="Embedding Images"):
        img_url = img.get("url")
        if img_url in seen_urls:
            continue
        seen_urls.add(img_url)

        try:
            response = requests.get(img_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")

            embedding = get_image_embedding(image)
            assert embedding.shape == (512,)  # Optional safety check

            image_vectors.append(embedding)
            image_metadata.append({
                "source": doc.metadata.get("source"),
                "title": doc.metadata.get("title"),
                "caption": img.get("alt", ""),
                "url": img_url,
            })
        except Exception as e:
            print(f"⚠️ Skipped image {img_url}: {e}")

    if image_vectors:
        # Convert to FAISS-compatible numpy array
        vector_array = np.array(image_vectors).astype("float32")

        # Build FAISS index
        index = faiss.IndexFlatL2(512)
        index.add(vector_array)

        # Save index and metadata
        faiss.write_index(index, "vectorstore/faiss_images.index")
        joblib.dump(image_metadata, "vectorstore/image_metadata.pkl")

        print(f"✅ Image vectorstore saved with {len(image_vectors)} vectors.")
    else:
        print("⚠️ No image vectors were created.")


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"]
    docs = load_documents()
    build_text_vectorstore(docs)
    build_image_vectorstore(docs)
