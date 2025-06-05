# 🧠 Multimodal RAG System – News QA from The Batch

This project implements a **Multimodal Retrieval-Augmented Generation (RAG) system** that answers user questions using news articles from [The Batch](https://www.deeplearning.ai/the-batch/), combining both **textual** and **visual** data (images). It retrieves relevant documents, performs question answering with GPT-4, and shows related images using CLIP-based vector search.

---

## 🚀 Features

- 🔍 Text-based search over news articles using OpenAI embeddings + FAISS
- 🧠 GPT-4 based answer generation using LangChain RAG chain
- 🖼️ Document-level image display (title, authors, alt text, URL)
- 🌐 Global image search using CLIP (text-to-image semantic similarity)
- 💬 Streamlit interface for easy query input and result visualization
- 🧪 Human evaluation and feedback form

## 🧱 Project Structure
batch-mrag-system/
├── app.py # Streamlit frontend
├── main.py # CLI version (console testing)
├── parser.py # Downloads and parses articles
├── vector_storer.py # Builds and stores text/image vectorstores
├── vector_loader.py # Loads FAISS text/image vectorstores
├── image_embedder.py # CLIP-based image & text embedders
├── rag_chain.py # RAG chain using LangChain
├── requirements.txt
├── data/
│ └── source_url.txt # URLs of The Batch articles
└── vectorstore/ # Saved FAISS indexes and metadata

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/lmidzhak/batch-mrag-system.git
```
### 2. Create and Activate a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Set Up Environment Variables
Create a .env file with OPENAI_API_KEY variable

### 5. Scrape Batch website for urls
```bash
python article_scraper.py
```
### 6. Create vectorstores
```bash
python vector_storer.py
```
### 7. Run the Streamlit App
```bash
python streamlit run app.py
```

