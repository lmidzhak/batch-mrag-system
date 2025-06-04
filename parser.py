from newspaper import Article, Config
from langchain.schema import Document


def load_documents():
    user_config = Config()
    user_config.request_timeout = 20

    # Load URLs from file
    with open('data/urls/source_url.txt', 'r') as file:
        urls = [url.strip() for url in file.readlines() if url.strip()]

    documents = []

    for url in urls:
        try:
            article = Article(url, config=user_config)
            article.download()
            article.parse()

            # Split article into paragraph chunks
            for chunk in article.text.split("\n\n"):
                chunk = chunk.strip()
                if chunk:
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source": url, "image_urls": list(article.images)}
                    ))

        except Exception as e:
            print(f"[SKIPPED] {url} due to error: {e}")

    print(f"✅ Parsed {len(documents)} chunks from {len(urls)} articles.")
    return documents
