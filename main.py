from dotenv import load_dotenv
from vector_loader import load_text_vectorstore, load_image_vectorstore
from rag_chain import get_rag_chain
from text_embedder import get_text_embedding

import numpy as np

load_dotenv()


def filter_relevant_images(image_metadata_list, query):
    query_keywords = set(query.lower().split())
    relevant_images = []

    for img_data in image_metadata_list:
        if isinstance(img_data, dict):
            alt_text = img_data.get("alt", "").lower()
            if any(keyword in alt_text for keyword in query_keywords):
                url = img_data.get("url")
                if url:
                    relevant_images.append(url)

    return relevant_images


def run_query(query: str):
    # Load vectorstores
    text_vectorstore = load_text_vectorstore()
    image_index, image_metadata = load_image_vectorstore()

    # Create RAG chain for text answers
    rag_chain = get_rag_chain(text_vectorstore)
    result = rag_chain.invoke({"query": query})

    print("\n🧠 GPT-4 Answer:\n")
    print(result["result"])

    print("\n📚 Sources with Metadata:\n")
    for i, doc in enumerate(result["source_documents"], 1):
        metadata = doc.metadata
        print(f"🔹 Source {i}")
        print(f"🔗 URL:         {metadata.get('source')}")
        print(f"📰 Title:       {metadata.get('title')}")
        print(f"✍️ Authors:     {', '.join(metadata.get('authors', []))}")
        print(f"📅 Published:   {metadata.get('publish_date')}")
        print(f"📄 Snippet:     {doc.page_content[:300].strip()}...")

        top_image = metadata.get("top_image")
        if top_image:
            print(f"🖼️ Top Image:   {top_image}")

        # Images attached to this doc
        image_urls = metadata.get("image_urls", [])
        relevant_images = filter_relevant_images(image_urls, query)

        if relevant_images:
            print("🖼️ Query-Relevant Images (From This Doc):")
            for img in relevant_images[:3]:
                print(f"   📷 {img}")
        else:
            print("🖼️ No relevant images found in doc metadata.")

    # 🔍 Global CLIP image search
    print("\n🌐 Globally Relevant Images via CLIP:\n")
    try:
        clip_query_vector = np.array([get_text_embedding(query)]).astype("float32")
        D, I = image_index.search(clip_query_vector, k=3)

        for rank, idx in enumerate(I[0], 1):
            img = image_metadata[idx]
            print(f"{rank}. 📷 {img['url']}")
            print(f"   📝 Caption: {img['caption']}")
            print(f"   🔗 Source:  {img['source']}")
            print(f"   📰 Title:   {img['title']}\n")
    except Exception as e:
        print(f"⚠️ Could not complete global image search: {e}")


if __name__ == "__main__":
    run_query("What do you know about Jeff Bezos?")
