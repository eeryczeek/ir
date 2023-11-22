import random
import time
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup


class WikiArticleGetter:
    def retrieve_single_wiki_article(self, link: str = None) -> None:
        time.sleep(random.uniform(0.1, 0.2))
        response = requests.get(
            link if link else "https://en.wikipedia.org/wiki/Special:Random")
        soup = BeautifulSoup(response.text, 'html.parser')

        main_content = soup.find("div", {"id": "mw-content-text"})

        if main_content:
            article_text = main_content.get_text()
            retrieved_link = re.search(r'Retrieved from "(.*?)"', article_text)

            if retrieved_link:
                article_text = article_text.replace(
                    retrieved_link.group(0), '')
                return retrieved_link.group(1), article_text
            else:
                raise Exception("Could not find the 'Retrieved from' link.")
        else:
            raise Exception("Could not find article content on this page.")

    def retrieve_wiki_articles(self, number_of_articles: int, article_links_list: list = None) -> None:
        dict_of_pages = {}
        if article_links_list:
            for link in article_links_list:
                try:
                    retrieved_link, article_text = self.retrieve_single_wiki_article(
                        link)
                    dict_of_pages[retrieved_link] = article_text
                except Exception as e:
                    print(f"Error: {e}")

        while len(dict_of_pages) < number_of_articles:
            try:
                retrieved_link, article_text = self.retrieve_single_wiki_article()
                dict_of_pages[retrieved_link] = article_text
            except Exception as e:
                print(f"Error: {e}")
        return dict_of_pages

    def save_wiki_articles(self, filename: str, dict_of_pages: dict) -> None:
        dataframe = pd.DataFrame.from_dict(dict_of_pages, orient='index')
        dataframe.to_csv(filename)
        return None
