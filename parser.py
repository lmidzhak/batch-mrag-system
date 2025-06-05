import os
import time
import random

from newspaper import Article, Config
from langchain.schema import Document

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from bs4 import BeautifulSoup


def parse_article(url, config):
    try:
        article = Article(url, config=config)
        article.download()

        # Sleep for bypassing 429 Too Many Requests error
        time.sleep(random.uniform(1.5, 3.0))

        html = article.html
        soup = BeautifulSoup(html, 'html.parser')
        alt_lookup = {
            img.get("src"): img.get("alt", "").strip()
            for img in soup.find_all("img") if img.get("src")
        }
        article.parse()

        # Filter only real URLs

        image_urls = [
            img_url for img_url in article.images
            if img_url.startswith("http://") or img_url.startswith("https://")
        ]

        alt_results = [
            {"url": img_url, "alt": alt_lookup.get(img_url, "No alt text")}
            for img_url in image_urls
        ]

        metadata = {
            "source": url,
            "title": article.title,
            "authors": article.authors,
            "publish_date": str(article.publish_date) if article.publish_date else None,
            "top_image": article.top_image,
            "image_urls": alt_results
        }

        chunks = [
            Document(page_content=chunk.strip(), metadata=metadata)
            for chunk in article.text.split("\n\n") if chunk.strip()
        ]

        return chunks
    except Exception as e:
        print(f"[SKIPPED] {url} due to error: {e}")
        return []


def load_documents():
    config = Config()
    config.request_timeout = 20

    with open('data/urls/source_url.txt', 'r') as file:
        urls = [url.strip() for url in file if url.strip()]

    documents = []
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
        futures = {executor.submit(parse_article, url, config): url for url in urls}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing articles"):
            documents.extend(future.result())

    print(f"✅ Parsed {len(documents)} chunks from {len(urls)} articles.")
    return documents
