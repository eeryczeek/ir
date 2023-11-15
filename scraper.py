import random
import time
import requests
import re
from bs4 import BeautifulSoup


class WikiArticleGetter:
    def __init__(self):
        self.dict_of_pages = {}

    def retrieve_single_wiki_article(self):
        time.sleep(random.uniform(0, 1))
        response = requests.get("http://en.wikipedia.org/wiki/Special:Random")
        soup = BeautifulSoup(response.text, 'html.parser')

        main_content = soup.find("div", {"id": "mw-content-text"})

        if main_content:
            article_text = main_content.get_text()
            retrieved_link = re.search(r'Retrieved from "(.*?)"', article_text)

            if retrieved_link:
                article_text = article_text.replace(
                    retrieved_link.group(0), '')
                self.dict_of_pages[retrieved_link.group(1)] = article_text
            else:
                raise Exception("Could not find the 'Retrieved from' link.")
        else:
            raise Exception("Could not find article content on this page.")

    def retrieve_wiki_articles(self, number_of_articles):
        while len(self.dict_of_pages) < number_of_articles:
            try:
                self.retrieve_single_wiki_article()
            except Exception as e:
                print(f"Error: {e}")
