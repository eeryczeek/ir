from scraper import WikiArticleGetter
from text_processor import Preprocessor
from utils import save_wiki_articles, load_wiki_articles
from evaluator import Evaluator
import os

if __name__ == "__main__":
    if not os.path.exists("wiki_articles.csv"):
        wikiScraper = WikiArticleGetter()
        wikiScraper.retrieve_wiki_articles(4)

        preprocessor = Preprocessor()
        preprocessor.process_articles(wikiScraper.dict_of_pages)

        save_wiki_articles("wiki_articles.csv", preprocessor.processed_articles)

    wiki_articles = load_wiki_articles("wiki_articles.csv")

    wikiScraper = WikiArticleGetter()
    wikiScraper.retrieve_wiki_articles(2)
    preprocessor = Preprocessor()
    preprocessor.process_articles(wikiScraper.dict_of_pages)
    query = preprocessor.processed_articles

    evaluator = Evaluator(query, wiki_articles)
    ranked_scrapped_articles = evaluator.evaluate()
    print(ranked_scrapped_articles)
