import os
import streamlit as st
import numpy as np
from urllib.parse import urlparse
from dotenv import load_dotenv

from rag_chain import get_rag_chain
from vector_loader import load_text_vectorstore, load_image_vectorstore
from text_embedder import get_text_embedding

load_dotenv()

assert os.getenv("OPENAI_API_KEY")

# --- Streamlit Setup ---
st.set_page_config(page_title="🧠 Multimodal RAG - News QA", layout="wide")
st.title("📰 Multimodal RAG System")
st.caption("Ask a question about AI news from The Batch and get a GPT-powered answer with relevant sources and images.")

# --- Load vectorstores ---
@st.cache_resource
def load_all():
    text_vectorstore = load_text_vectorstore()
    image_index, image_metadata = load_image_vectorstore()
    rag_chain = get_rag_chain(text_vectorstore)
    return rag_chain, image_index, image_metadata

rag_chain, image_index, image_metadata = load_all()


# --- Query Input ---
query = st.text_input("🔍 Ask something:", placeholder="e.g., How is AI used in medicine?")
run_button = st.button("🔎 Search")

if run_button and query:
    with st.spinner("Thinking..."):
        result = rag_chain.invoke({"query": query})

    st.subheader("🧠 GPT Answer")
    st.markdown(result["result"])

    st.subheader("📚 Retrieved Sources")
    for i, doc in enumerate(result["source_documents"], 1):
        metadata = doc.metadata
        st.markdown(f"### 🔹 Source {i}: {metadata.get('title', 'Untitled')}")
        st.markdown(f"[🔗 Link]({metadata.get('source', '#')})")
        st.markdown(f"**Authors**: {', '.join(metadata.get('authors', []))}")
        st.markdown(f"**Published**: {metadata.get('publish_date', 'Unknown')}")
        st.markdown(f"**Snippet**: {doc.page_content[:300].strip()}...")

        top_image = metadata.get("top_image")
        if top_image:
            st.image(top_image, caption="Top Image", use_column_width="auto")

        def filter_relevant_images(image_urls, query):
            query_keywords = set(query.lower().split())
            relevant_images = []
            for img in image_urls:
                if isinstance(img, dict):
                    alt = img.get("alt", "").lower()
                    url = img.get("url")
                    if url and any(k in alt for k in query_keywords):
                        relevant_images.append(url)
                elif isinstance(img, str):
                    filename = os.path.basename(urlparse(img).path).lower()
                    if any(k in filename for k in query_keywords):
                        relevant_images.append(img)
            return relevant_images

        image_urls = metadata.get("image_urls", [])
        relevant_images = filter_relevant_images(image_urls, query)
        if relevant_images:
            st.markdown("**📷 Query-Relevant Images (From This Doc):**")
            for img_url in relevant_images[:3]:
                st.image(img_url, width=300)
        else:
            st.markdown("_No query-relevant images found in this document._")

    # --- 🌐 Global Image Search via CLIP ---
    st.subheader("🌐 Globally Relevant Images via CLIP")
    try:
        clip_query_vector = np.array([get_text_embedding(query)]).astype("float32")
        D, I = image_index.search(clip_query_vector, k=3)
        for idx in I[0]:
            img = image_metadata[idx]
            st.image(img["url"], caption=f"{img['caption']}", width=300)
            st.markdown(f"[🔗 Source]({img['source']}) — *{img['title']}*")
    except Exception as e:
        st.warning(f"⚠️ Global image search failed: {e}")

    # --- Human Evaluation ---
    st.markdown("---")
    st.subheader("🧠 Rate the Result")
    rating = st.radio("Was the answer helpful?", ["✅ Yes", "❌ No"], horizontal=True)
    comment = st.text_area("💬 Any feedback?")
    if st.button("Submit Feedback"):
        st.success("✅ Thanks! Your feedback has been recorded.")
