import os
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

os.makedirs("data/urls", exist_ok=True)

BASE_URL = "https://www.deeplearning.ai"
HOME_PAGE = "https://www.deeplearning.ai/the-batch/"
URL_OUTPUT_TXT_PATH = "data/urls/source_url.txt"


def get_page_urls(page_soup: BeautifulSoup) -> list[str]:
    return [urljoin(BASE_URL, link["href"]) for link in page_soup.find_all('a') if "issue" in link["href"]]


def get_next_page(page_soup: BeautifulSoup) -> str:
    next_page = page_soup.select_one('div.justify-self-end > a')
    if next_page:
        return next_page["href"]


def get_urls() -> list[str]:
    page = requests.get(HOME_PAGE).content
    first_page_soup = BeautifulSoup(page, "html.parser")

    next_page = get_next_page(first_page_soup)

    all_urls = set(get_page_urls(first_page_soup))
    while next_page is not None:
        page = requests.get(urljoin(BASE_URL, next_page)).content
        soup = BeautifulSoup(page, "html.parser")
        all_urls.update(get_page_urls(soup))
        next_page = get_next_page(soup)
        all_urls = all_urls
    return all_urls


def write_urls(urls: list[str]) -> None:
    with open(URL_OUTPUT_TXT_PATH, "w", encoding="utf-8") as file:
        for url in urls:
            file.write(url + "\n")


def main(path) -> None:
    all_urls = get_urls()
    write_urls(all_urls)


if __name__ == "__main__":
    main(URL_OUTPUT_TXT_PATH)
